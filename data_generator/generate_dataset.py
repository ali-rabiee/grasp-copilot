from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Tuple

from .episode import Episode, OBJECT_LABELS, write_jsonl
from .oracle import POUR_AMOUNTS, OracleState, oracle_decide_tool, validate_tool_call
from .oracle_registry import ENV_REGISTRY, get_spec


def _deepcopy_memory(mem: Dict) -> Dict:
    out = {
        "n_interactions": int(mem["n_interactions"]),
        "past_dialogs": list(mem["past_dialogs"]),
        "candidates": list(mem["candidates"]),
        "last_tool_calls": list(mem["last_tool_calls"]),
        "excluded_obj_ids": list(mem.get("excluded_obj_ids") or []),
        "last_action": dict(mem.get("last_action") or {}),
    }
    # Optional-but-important training context: the last assistant prompt shown to the user
    # (including the choice list). The GUI runtime also maintains this field.
    last_prompt = mem.get("last_prompt")
    if isinstance(last_prompt, dict):
        out["last_prompt"] = dict(last_prompt)
    return out


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
    stats: Optional[Dict] = None,
    *,
    yes_p: float = 0.5,
    none_of_them_p: float = 0.2,
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
        or state.awaiting_amount_choice
        or state.awaiting_amount_confirmation
    )
    if (not must_respond) and rng.random() >= 0.6:
        if isinstance(stats, dict):
            stats["user_silent_steps"] = int(stats.get("user_silent_steps", 0)) + 1
        return

    def append_user(content: str) -> None:
        memory["past_dialogs"].append({"role": "user", "content": content})
        if isinstance(stats, dict):
            stats["user_replies_total"] = int(stats.get("user_replies_total", 0)) + 1
            key = str(content).strip().upper()
            if key == "YES":
                stats["user_replies_yes"] = int(stats.get("user_replies_yes", 0)) + 1
            elif key == "NO":
                stats["user_replies_no"] = int(stats.get("user_replies_no", 0)) + 1
            elif str(content).strip().lower() == "none of them":
                stats["user_replies_none_of_them"] = int(stats.get("user_replies_none_of_them", 0)) + 1
            elif key in {"APPROACH", "ALIGN_YAW"}:
                stats[f"user_replies_mode_{key.lower()}"] = int(stats.get(f"user_replies_mode_{key.lower()}", 0)) + 1

            ctx_type = str((state.last_prompt_context or {}).get("type") or "unknown")
            ctx_counts = stats.setdefault("user_replies_by_context", {})
            if isinstance(ctx_counts, dict):
                bucket = ctx_counts.setdefault(ctx_type, {})
                if isinstance(bucket, dict):
                    bucket[key] = int(bucket.get(key, 0)) + 1

    if ctx.get("type") in {
        "confirm", "help", "confirm_stack", "confirm_pour", "confirm_grab",
        "pitcher_acquisition", "non_top_redirect", "cup_full_redirect",
    }:
        obj_id = ctx.get("obj_id") or ctx.get("alt_obj_id")
        resp = "YES" if rng.random() < float(yes_p) else "NO"
        append_user(resp)
        if resp == "YES" and obj_id:
            state.pending_action_obj_id = obj_id
            state.selected_obj_id = obj_id
            action = (ctx.get("action") or "").upper()
            if action in {"APPROACH", "ALIGN_YAW", "STACK", "GRAB", "POUR"}:
                state.pending_mode = action
            # Redirect prompts replace the intended target.
            if ctx["type"] in {"non_top_redirect", "cup_full_redirect"}:
                state.intended_obj_id = obj_id
            else:
                state.intended_obj_id = obj_id
            state.last_declined_obj_id = None
        elif resp == "NO" and obj_id:
            state.last_declined_obj_id = obj_id
            state.pending_action_obj_id = None
            state.pending_mode = None
            state.awaiting_anything_else = True
            # Loop preventer: exclude the declined object from future candidate
            # offers so the simulator/user can't be re-offered the same target.
            ex = set(memory.get("excluded_obj_ids") or [])
            ex.add(str(obj_id))
            memory["excluded_obj_ids"] = sorted(ex)
        if ctx["type"] in {"confirm", "help", "confirm_stack", "confirm_pour", "confirm_grab"}:
            state.selected_obj_id = None
        if ctx["type"] in {"confirm", "confirm_stack", "confirm_pour", "confirm_grab", "pitcher_acquisition",
                            "non_top_redirect", "cup_full_redirect"}:
            state.awaiting_confirmation = False
        if ctx["type"] == "help":
            state.awaiting_help = False
    elif ctx.get("type") in {"intent_gate_candidates", "intent_gate_yaw", "intent_gate_stack", "intent_gate_pour"}:
        # Decide whether the oracle's inferred intent matches the hidden intent often,
        # but not always, to produce recovery branches.
        inferred_labels = set(ctx.get("labels") or [])
        inferred_obj_id = ctx.get("obj_id")
        intended = episode.intended_obj()

        resp = "YES" if rng.random() < float(yes_p) else "NO"
        append_user(resp)
        if resp == "YES":
            ctx_type = ctx.get("type")
            if ctx_type in {"intent_gate_candidates", "intent_gate_stack", "intent_gate_pour"}:
                state.awaiting_choice = True
                state.awaiting_intent_gate = False
                action = str(ctx.get("action") or "APPROACH").upper()
                if action not in {"APPROACH", "ALIGN_YAW", "STACK", "POUR", "GRAB"}:
                    action = "APPROACH"
                state.pending_mode = action
            else:
                # intent_gate_yaw: ask if user wants help aligning.
                state.awaiting_help = True
                state.awaiting_intent_gate = False
                state.pending_mode = "ALIGN_YAW"
                if inferred_obj_id:
                    state.selected_obj_id = inferred_obj_id
        else:
            state.awaiting_intent_gate = False
            state.awaiting_choice = False
            state.awaiting_help = False
            state.awaiting_confirmation = False
            state.awaiting_anything_else = True
    elif ctx.get("type") == "anything_else":
        # If user says NO, end the episode early (no "NOOP" tool exists).
        resp = "YES" if rng.random() < float(yes_p) else "NO"
        append_user(resp)
        if resp == "YES":
            # Only clear exclusions if the live candidate set is otherwise empty
            # (e.g., user said "None of them" repeatedly). Otherwise preserve
            # them so previously-declined targets stay out of the next round —
            # this is the key loop preventer.
            cur_candidates = list(memory.get("candidates") or [])
            excluded = set(memory.get("excluded_obj_ids") or [])
            live = [c for c in cur_candidates if c not in excluded]
            if not live:
                memory["excluded_obj_ids"] = []
            state.awaiting_mode_select = True
            state.awaiting_anything_else = False
        else:
            state.terminate_episode = True
            state.awaiting_anything_else = False
    elif ctx.get("type") == "mode_select":
        # Pick a mode aligned with hidden intent. Choices vary per env, surfaced as ctx["actions"].
        actions = list(ctx.get("actions") or ["APPROACH", "ALIGN_YAW"])
        intended = episode.intended_obj()
        cur = episode.gripper_hist[-1]

        if "STACK" in actions:
            prefer = "APPROACH" if cur.cell != intended.cell else "STACK"
        elif "POUR" in actions or "GRAB" in actions:
            pitcher = getattr(episode, "pitcher", None)
            pitcher_obj = pitcher() if callable(pitcher) else None
            if pitcher_obj is not None and not pitcher_obj.is_held and "GRAB" in actions:
                prefer = "GRAB"
            else:
                prefer = "APPROACH" if cur.cell != intended.cell else "POUR"
        else:
            prefer = "APPROACH" if cur.cell != intended.cell else "ALIGN_YAW"

        if rng.random() < 0.15:
            others = [a for a in actions if a != prefer]
            if others:
                prefer = rng.choice(others)
        if prefer not in actions:
            prefer = actions[0]
        append_user(prefer)
        state.pending_mode = prefer
        state.awaiting_mode_select = False
        state.awaiting_choice = True
    elif ctx.get("type") == "amount_choice":
        # Pouring: pick a bucket. Bias toward the episode's hidden ``intended_amount``
        # so episodes terminate, but inject occasional disagreement → recovery branches.
        intended_amount = getattr(episode, "intended_amount", None) or "HALF"
        amounts = list(ctx.get("amounts") or list(POUR_AMOUNTS))
        none_index = int(ctx.get("none_index") or (len(amounts) + 1))
        if rng.random() < float(none_of_them_p):
            append_user("None — don't pour")
            state.awaiting_amount_choice = False
            state.awaiting_anything_else = True
        else:
            pick = intended_amount if (intended_amount in amounts and rng.random() < 0.85) else rng.choice(amounts)
            append_user(pick)
            state.pending_amount = pick
            state.awaiting_amount_choice = False
            state.awaiting_amount_confirmation = True
    elif ctx.get("type") == "confirm_amount":
        # Confirm the chosen amount. YES → ready to fire POUR; NO → re-pick.
        amount = ctx.get("amount") or state.pending_amount
        resp = "YES" if rng.random() < float(yes_p) else "NO"
        append_user(resp)
        if resp == "YES":
            state.awaiting_amount_confirmation = False
            # Re-arm the pending action so the next decide() emits POUR with this amount.
            state.pending_action_obj_id = state.selected_obj_id or state.intended_obj_id
            state.pending_mode = "POUR"
            state.awaiting_confirmation = False
        else:
            state.awaiting_amount_confirmation = False
            state.awaiting_amount_choice = True
            state.pending_amount = None
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

        # Balance object selection vs "None of them":
        # - With probability none_of_them_p -> choose "None of them"
        # - Otherwise choose an object label.
        if labels and rng.random() < float(none_of_them_p):
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
            # Still often pick the intended object to keep episodes coherent.
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
        # Fallback: if it's a strict YES/NO style prompt, answer with YES/NO.
        # Match against the semantic label (after stripping "N) ") and use
        # equality — not substring — so options like "None — don't pour"
        # don't trip the YES/NO branch via "NO" ⊂ "None".
        labels = [c for c in tool_call["args"].get("choices", [])]
        semantics = [_strip_choice_label(l).strip().upper() for l in labels]
        is_yes_no = "YES" in semantics and "NO" in semantics
        if is_yes_no:
            resp = "YES" if rng.random() < 0.55 else "NO"
            append_user(resp)
        else:
            # Otherwise, respond with a semantic label if possible.
            choices = list(tool_call["args"].get("choices", []) or [])
            if choices:
                pick = _strip_choice_label(rng.choice(choices))
                append_user(pick)

    state.last_prompt_context = None


def _schema_validate_record(rec: Dict, *, env: Optional[str] = None) -> None:
    for k in ("episode_id", "objects", "gripper_hist", "memory", "user_state", "target_tool_call"):
        if k not in rec:
            raise ValueError(f"Missing key: {k}")
    if len(rec["gripper_hist"]) != 6:
        raise ValueError("gripper_hist must have length 6")
    validate_tool_call(rec["target_tool_call"], env=env)
    us = rec.get("user_state") or {}
    if not isinstance(us, dict) or us.get("mode") not in {"translation", "rotation", "gripper"}:
        raise ValueError("user_state.mode must be one of {translation, rotation, gripper}")


def _build_episode(env: str, rng: random.Random, episode_id: int, *, n_obj_min: int, n_obj_max: int, collision_p: float):
    """Construct the per-env Episode object with env-appropriate kwargs."""
    spec = get_spec(env)
    cls = spec.episode_cls
    if env == "reach_to_grasp_ycb":
        max_n = min(int(n_obj_max), len(OBJECT_LABELS))
        min_n = max(2, min(int(n_obj_min), max_n))
        n_obj = rng.randint(min_n, max_n)
        return cls(rng=rng, episode_id=episode_id, n_obj=n_obj, collision_p=collision_p)
    if env == "cube_stacking":
        from .episode_stacking import CUBE_LABELS
        max_n = min(int(n_obj_max), len(CUBE_LABELS))
        min_n = max(2, min(int(n_obj_min), max_n))
        n_cubes = rng.randint(min_n, max_n)
        return cls(rng=rng, episode_id=episode_id, n_cubes=n_cubes)
    if env == "pouring":
        # n_cups inferred internally; n_obj_min/max ignored here.
        return cls(rng=rng, episode_id=episode_id)
    raise ValueError(f"Unsupported env: {env}")


def generate(
    episodes: int,
    seed: int,
    *,
    env: str = "reach_to_grasp_ycb",
    n_obj_min: int = 2,
    n_obj_max: int = 10,
    collision_p: float = 0.15,
    candidate_max_dist: Optional[int] = None,
    user_yes_p: float = 0.5,
    user_none_of_them_p: float = 0.2,
) -> Tuple[List[Dict], Dict]:
    spec = get_spec(env)
    decide_fn = spec.decide_fn
    if candidate_max_dist is None:
        candidate_max_dist = spec.default_candidate_max_dist

    rng = random.Random(seed)
    records: List[Dict] = []
    # Seed tool_counts with the env's full skill set so the stats file always has them.
    from .oracle import ENV_SKILLS
    tool_counts: Dict[str, int] = {t: 0 for t in ENV_SKILLS[env]}
    reply_stats: Dict = {
        "user_replies_total": 0,
        "user_replies_yes": 0,
        "user_replies_no": 0,
        "user_replies_none_of_them": 0,
        "user_replies_mode_approach": 0,
        "user_replies_mode_align_yaw": 0,
        "user_replies_mode_stack": 0,
        "user_replies_mode_grab": 0,
        "user_replies_mode_pour": 0,
        "user_silent_steps": 0,
        "user_replies_by_context": {},
    }

    for episode_id in range(episodes):
        ep = _build_episode(env, rng, episode_id, n_obj_min=n_obj_min, n_obj_max=n_obj_max, collision_p=collision_p)
        state = OracleState(intended_obj_id=ep.intended_obj_id)
        memory: Dict = {
            "n_interactions": 0,
            "past_dialogs": [],
            "candidates": ep.gripper_candidates(max_dist=candidate_max_dist),
            "last_tool_calls": [],
            "excluded_obj_ids": [],
            "last_action": {},
            # Store the last prompt (kind/text/choices + optional context) so the next-step
            # decision is learnable without relying on hidden oracle state.
            "last_prompt": {},
        }

        # Hard cap on per-episode interactions as a backstop against any
        # logic loop that slips past the per-prompt loop preventers. ~12 turns
        # is well above the natural episode length (median 4-6) yet small
        # enough that pathological cycles can't dominate the dataset.
        max_interactions_per_episode = 12

        for t in range(ep.T):
            if state.terminate_episode:
                break
            if int(memory.get("n_interactions", 0)) >= max_interactions_per_episode:
                state.terminate_episode = True
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
                "env": env,
            }

            tool_call = decide_fn(
                record["objects"],
                record["gripper_hist"],
                memory,
                state,
                user_state=record["user_state"],
            )
            validate_tool_call(tool_call, env=env)
            record["target_tool_call"] = tool_call
            _schema_validate_record(record, env=env)
            records.append(record)

            tool_counts[tool_call["tool"]] = tool_counts.get(tool_call["tool"], 0) + 1

            # Update dialog and interaction counters.
            if tool_call["tool"] == "INTERACT":
                memory["n_interactions"] += 1
                memory["past_dialogs"].append({"role": "assistant", "content": tool_call["args"]["text"]})
                # Persist the full prompt + options (and oracle-provided context when available).
                # This matches what the interactive GUI keeps in memory.
                memory["last_prompt"] = {
                    "kind": tool_call["args"].get("kind"),
                    "text": tool_call["args"].get("text"),
                    "choices": list(tool_call["args"].get("choices") or []),
                    # Context is optional metadata that helps interpret user replies like "1"/"2"
                    # and disambiguate prompt types (confirm vs help vs candidate_choice).
                    "context": dict(state.last_prompt_context or {}),
                }

            _simulate_user_response(
                rng,
                tool_call,
                ep,
                memory,
                state,
                stats=reply_stats,
                yes_p=float(user_yes_p),
                none_of_them_p=float(user_none_of_them_p),
            )

            # Maintain a short history of tool calls for memory logging.
            memory["last_tool_calls"].append(tool_call["tool"])
            memory["last_tool_calls"] = memory["last_tool_calls"][-3:]

            # Apply tool effects then simulate teleop toward intent.
            ep.apply_tool(tool_call)
            motion_tools = {"APPROACH", "ALIGN_YAW", "STACK", "GRAB", "POUR", "RELEASE"}
            if tool_call["tool"] in motion_tools:
                la: Dict = {"tool": tool_call["tool"]}
                if "obj" in (tool_call.get("args") or {}):
                    la["obj"] = tool_call["args"]["obj"]
                if "amount" in (tool_call.get("args") or {}):
                    la["amount"] = tool_call["args"]["amount"]
                memory["last_action"] = la
            if t < ep.T - 1:
                # Treat an INTERACT as a conversational turn: the user answers
                # the assistant, then motion resumes on a later timestep.
                skip_user_motion = tool_call["tool"] == "INTERACT" or rng.random() < 0.85
                if not skip_user_motion:
                    ep.apply_user_motion()

            # Env-specific termination conditions.
            if env == "reach_to_grasp_ycb":
                intended = ep.get_obj(state.intended_obj_id)
                g = ep.gripper_hist[-1]
                if g.cell == intended.cell and g.yaw == intended.yaw:
                    break
            elif env == "cube_stacking":
                # Terminate as soon as the held cube has been stacked on the
                # intended base (regardless of whether the base was itself the
                # topper of a pre-existing stack — its own ``stacked_on`` may
                # already be non-None from scene init).
                if any(o.stacked_on == state.intended_obj_id for o in ep.objects):
                    break
                # Also terminate if nothing is held anymore (RELEASE or a stack
                # to a non-intended base): no recovery is possible without a held cube.
                if not any(o.is_held for o in ep.objects):
                    break
            elif env == "pouring":
                # Terminate after a successful POUR on the intended cup.
                la = memory.get("last_action") or {}
                if la.get("tool") == "POUR" and la.get("obj") == state.intended_obj_id:
                    break

        # Reset per-episode flags that should not leak; none currently.
        state.last_tool_calls.clear()

    stats = {"tool_distribution": tool_counts, "user_reply_distribution": reply_stats}
    return records, stats


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10000,
                    help="Number of episodes to generate (default: 10000).")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env", type=str, default="reach_to_grasp_ycb",
                    choices=sorted(ENV_REGISTRY.keys()),
                    help="Which environment's oracle/episode to use.")
    ap.add_argument("--n_obj_min", type=int, default=2)
    ap.add_argument("--n_obj_max", type=int, default=10)
    ap.add_argument("--collision_p", type=float, default=0.15)
    ap.add_argument("--candidate_max_dist", type=int, default=None,
                    help="Override env default candidate radius (Manhattan).")
    ap.add_argument(
        "--user_yes_p",
        type=float,
        default=0.5,
        help="Simulated user YES probability for YES/NO prompts (target ~0.5 for balanced YES/NO).",
    )
    ap.add_argument(
        "--user_none_of_them_p",
        type=float,
        default=0.2,
        help="Simulated user probability of choosing 'None of them' on candidate-choice prompts (target ~0.2 for 80/20 object vs none).",
    )
    args = ap.parse_args(argv)

    records, stats = generate(
        episodes=args.episodes,
        seed=args.seed,
        env=args.env,
        n_obj_min=args.n_obj_min,
        n_obj_max=args.n_obj_max,
        collision_p=args.collision_p,
        candidate_max_dist=args.candidate_max_dist,
        user_yes_p=float(args.user_yes_p),
        user_none_of_them_p=float(args.user_none_of_them_p),
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_jsonl(args.out, records)
    with open(args.out + ".stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
