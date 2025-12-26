from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Tuple

from .episode import Episode, OBJECT_LABELS, write_jsonl
from .oracle import OracleState, oracle_decide_tool, validate_tool_call


def _deepcopy_memory(mem: Dict) -> Dict:
    return {
        "n_interactions": int(mem["n_interactions"]),
        "past_dialogs": list(mem["past_dialogs"]),
        "candidates": list(mem["candidates"]),
        "last_tool_calls": list(mem["last_tool_calls"]),
        "excluded_obj_ids": list(mem.get("excluded_obj_ids") or []),
        "last_action": dict(mem.get("last_action") or {}),
    }


def _infer_user_mode_from_gripper_hist(gripper_hist: List[Dict]) -> str:
    """
    Infer an input mode from the last delta in gripper history.
    Returns one of: translation | rotation | gripper
    """
    if len(gripper_hist) < 2:
        return "translation"
    a, b = gripper_hist[-2], gripper_hist[-1]
    if a.get("cell") != b.get("cell"):
        return "translation"
    if a.get("yaw") != b.get("yaw"):
        return "rotation"
    if a.get("z") != b.get("z"):
        return "gripper"
    return "translation"


def _strip_choice_label(choice: str) -> str:
    parts = choice.split(")", 1)
    return parts[1].strip() if len(parts) == 2 else choice.strip()


def _choice_index(choice: str) -> Optional[int]:
    """
    Extract the leading integer in a choice string like '2) foo' -> 2.
    """
    parts = choice.split(")", 1)
    if not parts:
        return None
    prefix = parts[0].strip()
    return int(prefix) if prefix.isdigit() else None


def _resolve_numbered_choice(user_text: str, choices: List[str]) -> Optional[str]:
    """
    If user_text is a number like '2', map it to the corresponding choice label (stripped).
    """
    t = user_text.strip()
    if not t.isdigit():
        return None
    idx = int(t)
    for c in choices:
        c_idx = _choice_index(c)
        if c_idx == idx:
            return _strip_choice_label(c)
    return None


def _simulate_user_response(
    rng: random.Random,
    tool_call: Dict,
    episode: Episode,
    memory: Dict,
    state: OracleState,
) -> None:
    """
    Apply the user-simulation rules to update memory and oracle state.
    """
    if tool_call["tool"] != "INTERACT":
        state.last_prompt_context = None
        return

    ctx = state.last_prompt_context or {}
    # If the oracle is explicitly waiting for a reply, always respond.
    # Otherwise, allow the simulated user to be occasionally quiet.
    must_respond = bool(
        state.awaiting_choice
        or state.awaiting_confirmation
        or state.awaiting_help
        or state.awaiting_intent_gate
        or state.awaiting_anything_else
        or state.awaiting_mode_select
    )
    if (not must_respond) and rng.random() >= 0.6:
        return

    def append_user(content: str) -> None:
        memory["past_dialogs"].append({"role": "user", "content": content})

    if ctx.get("type") in {"confirm", "help"}:
        obj_id = ctx.get("obj_id")
        aligns = obj_id == episode.intended_obj_id
        yes_prob = 0.75 if aligns else 0.25
        resp = "YES" if rng.random() < yes_prob else "NO"
        append_user(resp)
        if resp == "YES" and obj_id:
            state.pending_action_obj_id = obj_id
            state.selected_obj_id = obj_id
            # Lock mode to whatever was being confirmed (if present).
            action = (ctx.get("action") or "").upper()
            if action in {"APPROACH", "ALIGN_YAW"}:
                state.pending_mode = action
            # If the user confirmed, lock the goal to that object.
            state.intended_obj_id = obj_id
            state.last_declined_obj_id = None
        elif resp == "NO" and obj_id:
            state.last_declined_obj_id = obj_id
            # On NO, fall back into recovery / mode selection rather than acting.
            state.pending_action_obj_id = None
            state.pending_mode = None
            # If user rejected help/confirm, ask if anything else we can do.
            state.awaiting_anything_else = True
        # Once the user answered, don't keep re-confirming the same selection.
        # The oracle can rely on pending_action_obj_id (on YES) to proceed.
        if ctx["type"] in {"confirm", "help"}:
            state.selected_obj_id = None
        if ctx["type"] == "confirm":
            state.awaiting_confirmation = False
        if ctx["type"] == "help":
            state.awaiting_help = False
    elif ctx.get("type") in {"intent_gate_candidates", "intent_gate_yaw"}:
        # Decide whether the oracle's inferred intent matches the hidden intent often,
        # but not always, to produce recovery branches.
        inferred_labels = set(ctx.get("labels") or [])
        inferred_obj_id = ctx.get("obj_id")
        intended = episode.intended_obj()

        if inferred_obj_id:
            aligns = inferred_obj_id == intended.id
        else:
            aligns = bool(inferred_labels) and intended.label in inferred_labels

        yes_prob = 0.8 if aligns else 0.35
        resp = "YES" if rng.random() < yes_prob else "NO"
        append_user(resp)
        if resp == "YES":
            # Move to the next prompt: either object choice (candidates) or help confirm (yaw).
            if ctx.get("type") == "intent_gate_candidates":
                state.awaiting_choice = True
                state.awaiting_intent_gate = False
                # Set pending mode based on the prompt's implied action (translation vs rotation).
                action = str(ctx.get("action") or "APPROACH").upper()
                state.pending_mode = action if action in {"APPROACH", "ALIGN_YAW"} else "APPROACH"
            else:
                # Yaw struggle intent accepted: ask if user wants help.
                state.awaiting_help = True
                state.awaiting_intent_gate = False
                state.pending_mode = "ALIGN_YAW"
                if inferred_obj_id:
                    state.selected_obj_id = inferred_obj_id
        else:
            # Recovery: ask if anything else we can help with.
            state.awaiting_intent_gate = False
            state.awaiting_choice = False
            state.awaiting_help = False
            state.awaiting_confirmation = False
            state.awaiting_anything_else = True
    elif ctx.get("type") == "anything_else":
        # If user says NO, end the episode early (no "NOOP" tool exists).
        resp = "YES" if rng.random() < 0.75 else "NO"
        append_user(resp)
        if resp == "YES":
            # If the user said "None of them" repeatedly, they may have excluded all nearby objects.
            # Clear exclusions to let the assistant restart the help flow cleanly.
            memory["excluded_obj_ids"] = []
            state.awaiting_mode_select = True
            state.awaiting_anything_else = False
        else:
            state.terminate_episode = True
            state.awaiting_anything_else = False
    elif ctx.get("type") == "mode_select":
        # Pick a mode aligned with the hidden intent: if we're not in the target cell yet,
        # prefer approach; otherwise, align yaw.
        intended = episode.intended_obj()
        cur = episode.gripper_hist[-1]
        prefer = "APPROACH" if cur.cell != intended.cell else "ALIGN_YAW"
        # Allow occasional mismatch to diversify data.
        if rng.random() < 0.15:
            prefer = "ALIGN_YAW" if prefer == "APPROACH" else "APPROACH"
        # Reply with semantic label (preferred for training).
        append_user(prefer)
        state.pending_mode = prefer
        state.awaiting_mode_select = False
        state.awaiting_choice = True
    elif ctx.get("type") == "candidate_choice":
        # Choose which object; respond with the semantic label (preferred for training).
        labels = list(ctx.get("labels") or [])
        obj_ids = list(ctx.get("obj_ids") or [])
        none_index = int(ctx.get("none_index") or (len(labels) + 1))
        intended_label = episode.intended_obj().label
        declined_label: Optional[str] = None
        if state.last_declined_obj_id is not None:
            for o in episode.objects:
                if o.id == state.last_declined_obj_id:
                    declined_label = o.label
                    break

        # If the intended label is not in the current options (or we want to diversify),
        # select "None of them" sometimes, and exclude these ids going forward.
        if (labels and intended_label not in labels and rng.random() < 0.75) or (labels and rng.random() < 0.12):
            append_user("None of them")
            ex = set(memory.get("excluded_obj_ids") or [])
            for oid in obj_ids:
                ex.add(oid)
            memory["excluded_obj_ids"] = sorted(ex)
            state.awaiting_choice = True
            state.awaiting_confirmation = False
            state.selected_obj_id = None
            state.last_prompt_context = None
            return

        pick_label: Optional[str] = None
        if labels and intended_label in labels and rng.random() < 0.8:
            if declined_label is not None and intended_label == declined_label and len(labels) >= 2 and rng.random() < 0.85:
                others = [l for l in labels if l != declined_label]
                pick_label = rng.choice(others) if others else intended_label
            else:
                pick_label = intended_label
        elif labels:
            pick_label = rng.choice(labels)
        else:
            pick_label = _strip_choice_label(rng.choice(tool_call["args"]["choices"]))

        append_user(str(pick_label))

        # Map to object and update oracle state.
        for o in episode.objects:
            if o.label == pick_label:
                state.selected_obj_id = o.id
                state.intended_obj_id = o.id
                break
        state.awaiting_choice = False
        state.awaiting_confirmation = False
    else:
        # Fallback: if it's a YES/NO style prompt, answer with YES/NO.
        labels = [c for c in tool_call["args"].get("choices", [])]
        if any("YES" in l.upper() for l in labels) and any("NO" in l.upper() for l in labels):
            resp = "YES" if rng.random() < 0.55 else "NO"
            append_user(resp)
        else:
            # Otherwise, respond with a semantic label if possible.
            choices = list(tool_call["args"].get("choices", []) or [])
            if choices:
                pick = _strip_choice_label(rng.choice(choices))
                append_user(pick)

    state.last_prompt_context = None


def _schema_validate_record(rec: Dict) -> None:
    for k in ("episode_id", "objects", "gripper_hist", "memory", "user_state", "target_tool_call"):
        if k not in rec:
            raise ValueError(f"Missing key: {k}")
    if len(rec["gripper_hist"]) != 6:
        raise ValueError("gripper_hist must have length 6")
    validate_tool_call(rec["target_tool_call"])
    us = rec.get("user_state") or {}
    if not isinstance(us, dict) or us.get("mode") not in {"translation", "rotation", "gripper"}:
        raise ValueError("user_state.mode must be one of {translation, rotation, gripper}")


def generate(
    episodes: int,
    seed: int,
    *,
    n_obj_min: int = 2,
    n_obj_max: int = 10,
    collision_p: float = 0.15,
    candidate_max_dist: int = 1,
) -> Tuple[List[Dict], Dict]:
    rng = random.Random(seed)
    records: List[Dict] = []
    tool_counts: Dict[str, int] = {"INTERACT": 0, "APPROACH": 0, "ALIGN_YAW": 0}

    for episode_id in range(episodes):
        max_n = min(int(n_obj_max), len(OBJECT_LABELS))
        min_n = max(2, min(int(n_obj_min), max_n))
        n_obj = rng.randint(min_n, max_n)
        ep = Episode(rng=rng, episode_id=episode_id, n_obj=n_obj, collision_p=collision_p)
        state = OracleState(intended_obj_id=ep.intended_obj_id)
        memory: Dict = {
            "n_interactions": 0,
            "past_dialogs": [],
            "candidates": ep.gripper_candidates(max_dist=candidate_max_dist),
            "last_tool_calls": [],
            "excluded_obj_ids": [],
            "last_action": {},
        }

        for t in range(ep.T):
            if state.terminate_episode:
                break
            # Snapshot before choosing the tool call.
            memory["candidates"] = ep.gripper_candidates(max_dist=candidate_max_dist)
            gripper_hist = [p.to_record() for p in ep.gripper_hist]
            record = {
                "episode_id": episode_id,
                "objects": [o.to_record() for o in ep.objects],
                "gripper_hist": gripper_hist,
                "memory": _deepcopy_memory(memory),
                "user_state": {"mode": _infer_user_mode_from_gripper_hist(gripper_hist)},
            }

            tool_call = oracle_decide_tool(
                record["objects"],
                record["gripper_hist"],
                memory,
                state,
                user_state=record["user_state"],
            )
            validate_tool_call(tool_call)
            record["target_tool_call"] = tool_call
            _schema_validate_record(record)
            records.append(record)

            tool_counts[tool_call["tool"]] += 1

            # Update dialog and interaction counters.
            if tool_call["tool"] == "INTERACT":
                memory["n_interactions"] += 1
                memory["past_dialogs"].append({"role": "assistant", "content": tool_call["args"]["text"]})

            _simulate_user_response(rng, tool_call, ep, memory, state)

            # Maintain a short history of tool calls for memory logging.
            memory["last_tool_calls"].append(tool_call["tool"])
            memory["last_tool_calls"] = memory["last_tool_calls"][-3:]

            # Apply tool effects then simulate teleop toward intent.
            ep.apply_tool(tool_call)
            if tool_call["tool"] in {"APPROACH", "ALIGN_YAW"}:
                memory["last_action"] = {"tool": tool_call["tool"], "obj": tool_call["args"]["obj"]}
            if t < ep.T - 1:
                # If the assistant executed a non-interactive tool, the human is less likely
                # to keep fighting the motion on the very next step. This reduces unrealistic
                # "ALIGN_YAW spam" / oscillatory behavior.
                skip_user_motion = tool_call["tool"] != "INTERACT" and rng.random() < 0.85
                if not skip_user_motion:
                    ep.apply_user_motion()

            # If we've reached the currently intended object pose (cell + yaw),
            # stop the episode early. This makes APPROACH/ALIGN_YAW naturally "final"
            # actions and avoids long post-goal chat loops.
            intended = ep.get_obj(state.intended_obj_id)
            g = ep.gripper_hist[-1]
            if g.cell == intended.cell and g.yaw == intended.yaw:
                break

        # Reset per-episode flags that should not leak; none currently.
        state.last_tool_calls.clear()

    stats = {"tool_distribution": tool_counts}
    return records, stats


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_obj_min", type=int, default=2)
    ap.add_argument("--n_obj_max", type=int, default=10)
    ap.add_argument("--collision_p", type=float, default=0.15)
    ap.add_argument("--candidate_max_dist", type=int, default=1)
    args = ap.parse_args(argv)

    records, stats = generate(
        episodes=args.episodes,
        seed=args.seed,
        n_obj_min=args.n_obj_min,
        n_obj_max=args.n_obj_max,
        collision_p=args.collision_p,
        candidate_max_dist=args.candidate_max_dist,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_jsonl(args.out, records)
    with open(args.out + ".stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()

