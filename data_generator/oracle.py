from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import grid
from . import yaw as yawlib


def _normalize_label(text: str) -> str:
    # Accept underscores/spaces and case-insensitive matches.
    return text.strip().lower().replace(" ", "_")


def _extract_user_label(dialog: Sequence[Dict], valid_labels: Sequence[str]) -> Optional[str]:
    valid = {_normalize_label(l): l for l in valid_labels}
    # Search most recent user messages first.
    for msg in reversed(dialog):
        if msg.get("role") != "user":
            continue
        norm = _normalize_label(msg.get("content", ""))
        if norm in valid:
            return valid[norm]
    return None


@dataclass
class OracleState:
    selected_obj_id: Optional[str] = None
    just_guide: bool = False
    last_user_selection_label: Optional[str] = None
    pending_user_prompt: Optional[str] = None  # "ambiguity" or "takeover"
    pending_ambiguity_choices: Optional[Tuple[str, str]] = None  # labels
    pending_takeover_obj_id: Optional[str] = None
    # After the user accepts takeover, suppress re-offering takeover for a few steps
    # so the oracle proceeds with action tool calls (matches the spec's intent).
    takeover_cooldown_steps: int = 0
    outcomes: List[str] = field(default_factory=list)  # previous outcomes (t-1 ...), includes "none"


def _maybe_apply_takeover_reply(dialog: Sequence[Dict], state: OracleState) -> None:
    """
    Detect a user reply to the most recent takeover offer and update state.

    Generator uses fixed strings ("yes please" / "no, just guide"), but we accept
    simple substrings to keep it robust and deterministic.
    """
    if len(dialog) < 2:
        return
    last, prev = dialog[-1], dialog[-2]
    if last.get("role") != "user" or prev.get("role") != "assistant":
        return
    prev_txt = str(prev.get("content", "")).lower()
    if "take over" not in prev_txt and "takeover" not in prev_txt:
        return
    user_txt = str(last.get("content", "")).strip().lower()
    if "no" in user_txt:
        state.just_guide = True
        return
    if "yes" in user_txt:
        state.just_guide = False
        state.takeover_cooldown_steps = max(state.takeover_cooldown_steps, 3)


def _score_objects(obs: Dict) -> Dict[str, int]:
    objects = [o for o in obs["objects"] if not o["is_held"]]
    candidates = set(obs["candidates"])
    objects = [o for o in objects if o["id"] in candidates]

    recent_cell = obs["gripper_hist"][-1]["cell"]
    prev_cell = obs["gripper_hist"][-2]["cell"]

    # Pre-compute 2 nearest by grid distance (ties broken deterministically by obj_id).
    dist_items = [(o["id"], grid.manhattan(recent_cell, o["cell"])) for o in objects]
    dist_items.sort(key=lambda x: (x[1], x[0]))
    nearest2 = {oid for oid, _ in dist_items[:2]}

    scores: Dict[str, int] = {o["id"]: 0 for o in objects}
    for o in objects:
        oid = o["id"]
        if o["cell"] == recent_cell:
            scores[oid] += 2
        if o["cell"] == prev_cell:
            scores[oid] += 1
        if grid.same_row_or_col(o["cell"], recent_cell):
            scores[oid] += 1
        if oid in nearest2:
            scores[oid] += 1
    return scores


def _top2(scores: Dict[str, int]) -> List[Tuple[str, int]]:
    items = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return items[:2]


def _tool(tool_name: str, arguments: Dict) -> Dict:
    return {"tool_name": tool_name, "arguments": arguments}


def _interact(type_: str, text: str, choices: Optional[List[str]] = None) -> Dict:
    args: Dict = {"type": type_, "text": text}
    if choices is not None:
        args["choices"] = choices
    return _tool("INTERACT", args)


def oracle_decide_tool(obs: Dict, dialog: Sequence[Dict], state: OracleState) -> Dict:
    # If user previously requested "take over", exit just-guide mode.
    for msg in reversed(dialog):
        if msg.get("role") != "user":
            continue
        txt = msg.get("content", "").lower()
        if "take over" in txt or "takeover" in txt:
            state.just_guide = False
            break

    # If user just responded to a takeover offer, reflect that here.
    _maybe_apply_takeover_reply(dialog, state)
    if state.takeover_cooldown_steps > 0:
        # Consume one step immediately so it applies to this decision too.
        state.takeover_cooldown_steps -= 1

    # "Just guide" episodes: only coach unless user asked to take over.
    if state.just_guide:
        scores = _score_objects(obs)
        top = _top2(scores)
        hint = "I can guide you—move toward the likely target."
        if top:
            obj_id, _ = top[0]
            label = next(o["label"] for o in obs["objects"] if o["id"] == obj_id)
            cell = next(o["cell"] for o in obs["objects"] if o["id"] == obj_id)
            hint = f"I think it's the {label}; move the gripper to {cell} and lower to LOW."
        return _interact("coach", hint)

    # Direct user selection override (only when it's new or changes the current selection).
    user_label = _extract_user_label(dialog, [o["label"] for o in obs["objects"]])
    if user_label is not None and user_label != state.last_user_selection_label:
        state.last_user_selection_label = user_label
        for o in obs["objects"]:
            if not o["is_held"] and o["label"] == user_label and o["id"] in set(obs["candidates"]):
                if state.selected_obj_id != o["id"]:
                    state.selected_obj_id = o["id"]
                    return _tool("SELECT_TARGET", {"obj_id": o["id"]})

    # Rule 1.
    if obs["last_action_outcome"] == "grasp_success":
        # Clear selection if it's now held.
        if state.selected_obj_id is not None:
            held = next((o for o in obs["objects"] if o["id"] == state.selected_obj_id), None)
            if held and held["is_held"]:
                state.selected_obj_id = None
        return _interact("notify", "Got it.")

    scores = _score_objects(obs)
    top2 = _top2(scores)
    candidates = set(obs["candidates"])

    # Rule 2: ambiguity.
    if state.selected_obj_id is None and len(top2) == 2:
        (a_id, a_s), (b_id, b_s) = top2
        if abs(a_s - b_s) <= 1 and a_id in candidates and b_id in candidates:
            a_label = next(o["label"] for o in obs["objects"] if o["id"] == a_id)
            b_label = next(o["label"] for o in obs["objects"] if o["id"] == b_id)
            return _interact(
                "question",
                "Which object should I help with?",
                choices=[a_label, b_label],
            )

    # Rule 3: struggle detection.
    gh = obs["gripper_hist"]
    if len(gh) >= 3:
        same_cell_3 = gh[-1]["cell"] == gh[-2]["cell"] == gh[-3]["cell"]
        yaw_changed_twice = gh[-1]["yaw_bin"] != gh[-2]["yaw_bin"] and gh[-2]["yaw_bin"] != gh[-3]["yaw_bin"]
        oscillating = same_cell_3 and yaw_changed_twice
    else:
        oscillating = False

    recent_outcomes = list(state.outcomes[-5:])
    # state.outcomes includes the previous step outcome; ensure we count current obs outcome too.
    if not recent_outcomes or recent_outcomes[-1] != obs["last_action_outcome"]:
        recent_outcomes.append(obs["last_action_outcome"])
    fail_count = sum(1 for o in recent_outcomes if o in {"missed_contact", "grasp_fail"})
    if (oscillating or fail_count >= 2) and state.takeover_cooldown_steps == 0:
        return _interact("offer_takeover", "Want me to align yaw / take over?")

    # Rule 4: select if none.
    if state.selected_obj_id is None or state.selected_obj_id not in candidates:
        if not top2:
            return _interact("notify", "No candidates available.")
        state.selected_obj_id = top2[0][0]
        return _tool("SELECT_TARGET", {"obj_id": state.selected_obj_id})

    target_id = state.selected_obj_id
    target = next(o for o in obs["objects"] if o["id"] == target_id)
    gcur = obs["gripper_hist"][-1]

    # Rule 5: approach.
    if gcur["cell"] != target["cell"]:
        return _tool("APPROACH", {"obj_id": target_id, "mode": "topdown", "hover": "HIGH"})

    # Rule 6: align yaw.
    if gcur["yaw_bin"] != target["yaw_bin"]:
        return _tool("ALIGN_YAW", {"obj_id": target_id, "yaw_bin": target["yaw_bin"]})

    # Rule 7: grasp if ready.
    if gcur["z"] == "LOW":
        return _tool("GRASP", {"obj_id": target_id, "grasp_type": "topdown_pincher"})

    # Rule 8: next best suggestion / recovery meta-call.
    if obs["last_action_outcome"] in {"missed_contact", "grasp_fail"}:
        # If we already seem aligned but are failing, emit a recovery policy tool call.
        policy = "retry"
        if gcur["cell"] != target["cell"]:
            policy = "reapproach"
        elif gcur["yaw_bin"] != target["yaw_bin"]:
            policy = "realign"
        return _tool("RETRY_OR_ABORT", {"policy": policy, "obj_id": target_id})

    return _interact("coach", "Lower to LOW over the target, then grasp.")


def validate_tool_call(tool_call: Dict) -> None:
    if not isinstance(tool_call, dict) or set(tool_call.keys()) != {"tool_name", "arguments"}:
        raise ValueError("Tool call must be {tool_name, arguments}")
    name = tool_call["tool_name"]
    args = tool_call["arguments"]
    if name not in {"INTERACT", "SELECT_TARGET", "APPROACH", "ALIGN_YAW", "GRASP", "RETRY_OR_ABORT"}:
        raise ValueError(f"Invalid tool_name: {name}")
    if not isinstance(args, dict):
        raise ValueError("arguments must be an object")
    if name == "INTERACT":
        if set(args.keys()) not in ({"type", "text"}, {"type", "text", "choices"}):
            raise ValueError("INTERACT args must be {type,text[,choices]}")
        if args["type"] not in {"question", "confirm", "coach", "offer_takeover", "notify"}:
            raise ValueError("Invalid INTERACT.type")
        if not isinstance(args["text"], str):
            raise ValueError("Invalid INTERACT.text")
        if "choices" in args:
            if args["type"] != "question":
                raise ValueError("choices only allowed for type=question")
            if not isinstance(args["choices"], list) or not all(isinstance(x, str) for x in args["choices"]):
                raise ValueError("Invalid INTERACT.choices")
    elif name == "SELECT_TARGET":
        if set(args.keys()) != {"obj_id"} or not isinstance(args["obj_id"], str):
            raise ValueError("SELECT_TARGET args must be {obj_id}")
    elif name == "APPROACH":
        if set(args.keys()) != {"obj_id", "mode", "hover"}:
            raise ValueError("APPROACH args must be {obj_id,mode,hover}")
        if args["mode"] != "topdown":
            raise ValueError("APPROACH.mode must be topdown")
        if args["hover"] not in {"HIGH", "MID"}:
            raise ValueError("APPROACH.hover must be HIGH|MID")
    elif name == "ALIGN_YAW":
        if set(args.keys()) != {"obj_id", "yaw_bin"}:
            raise ValueError("ALIGN_YAW args must be {obj_id,yaw_bin}")
        if args["yaw_bin"] not in set(yawlib.YAW_BINS):
            raise ValueError("Invalid ALIGN_YAW.yaw_bin")
    elif name == "GRASP":
        if set(args.keys()) != {"obj_id", "grasp_type"}:
            raise ValueError("GRASP args must be {obj_id,grasp_type}")
        if args["grasp_type"] != "topdown_pincher":
            raise ValueError("Invalid GRASP.grasp_type")
    elif name == "RETRY_OR_ABORT":
        if set(args.keys()) != {"policy", "obj_id"}:
            raise ValueError("RETRY_OR_ABORT args must be {policy,obj_id}")
        if args["policy"] not in {"retry", "realign", "reapproach", "ask_user", "abort"}:
            raise ValueError("Invalid RETRY_OR_ABORT.policy")


