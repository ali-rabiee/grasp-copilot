from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import grid
from . import yaw as yawlib

UserState = Dict[str, str]  # expects {"mode": "translation"|"rotation"|"gripper"}

@dataclass
class OracleState:
    intended_obj_id: str
    selected_obj_id: Optional[str] = None
    pending_action_obj_id: Optional[str] = None
    pending_mode: Optional[str] = None  # "APPROACH" | "ALIGN_YAW"
    awaiting_confirmation: bool = False
    awaiting_help: bool = False
    awaiting_choice: bool = False
    awaiting_intent_gate: bool = False
    awaiting_anything_else: bool = False
    awaiting_mode_select: bool = False
    last_prompt_context: Optional[Dict] = None
    # Track recent user rejections to avoid asking the exact same question in a loop.
    last_declined_obj_id: Optional[str] = None
    last_tool_calls: List[str] = field(default_factory=list)
    terminate_episode: bool = False


def _tool(tool: str, args: Dict) -> Dict:
    return {"tool": tool, "args": args}


def _interact(kind: str, text: str, choices: List[str], context: Dict, state: OracleState) -> Dict:
    state.last_prompt_context = context
    return _tool("INTERACT", {"kind": kind, "text": text, "choices": choices})


def _rank_candidates(objects: Sequence[Dict], candidates: Sequence[str], gripper_cell: str) -> List[Dict]:
    available = [o for o in objects if o["id"] in set(candidates) and not o["is_held"]]
    scored = [(o, grid.manhattan(gripper_cell, o["cell"])) for o in available]
    scored.sort(key=lambda x: (x[1], x[0]["id"]))
    return [o for o, _ in scored]


def _top_two_candidates(objects: Sequence[Dict], candidates: Sequence[str], gripper_cell: str) -> Optional[List[Dict]]:
    ranked = _rank_candidates(objects, candidates, gripper_cell)
    if len(ranked) < 2:
        return None
    return [ranked[0], ranked[1]]


def _has_yaw_oscillation(gripper_hist: Sequence[Dict]) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """
    Returns (triggered, cell, yaw1, yaw2) to support yaw-struggle prompts.
    """
    if len(gripper_hist) < 6:
        return False, None, None, None
    cells = [g["cell"] for g in gripper_hist]
    yaws = [g["yaw"] for g in gripper_hist]
    cell_counts = Counter(cells)
    dominant_cell, count = cell_counts.most_common(1)[0]
    if count < 4:
        return False, None, None, None

    # Track yaw switches to see if we bounce between two bins.
    unique_order: List[str] = []
    for y in yaws:
        if not unique_order or unique_order[-1] != y:
            unique_order.append(y)
    if len(set(unique_order)) != 2:
        return False, None, None, None
    switches = sum(1 for i in range(1, len(yaws)) if yaws[i] != yaws[i - 1])
    if switches < 3:
        return False, None, None, None
    yaw1, yaw2 = unique_order[0], unique_order[1]
    return True, dominant_cell, yaw1, yaw2


def _has_cell_oscillation(gripper_hist: Sequence[Dict], cell_a: str, cell_b: str) -> bool:
    """
    Detects whether the gripper has been oscillating between two cells recently.
    This supports ambiguity prompts even when last_tool_calls is empty.
    """
    if len(gripper_hist) < 6:
        return False
    cells = [g["cell"] for g in gripper_hist]
    allowed = {cell_a, cell_b}
    # Require both cells to appear at least twice.
    if cells.count(cell_a) < 2 or cells.count(cell_b) < 2:
        return False
    # Require multiple transitions between the two cells.
    transitions = 0
    for i in range(1, len(cells)):
        if cells[i] != cells[i - 1] and {cells[i], cells[i - 1]} <= allowed:
            transitions += 1
    return transitions >= 2


def _effective_mode(user_state: Optional[UserState], gripper_hist: Sequence[Dict], memory: Dict) -> str:
    """
    Determine the effective user input mode.
    - translation -> approach-focused prompts
    - rotation -> align-focused prompts
    - gripper -> pseudo-random choice (deterministic, no RNG)
    """
    mode = str((user_state or {}).get("mode") or "translation")
    if mode not in {"translation", "rotation", "gripper"}:
        mode = "translation"
    if mode != "gripper":
        return mode
    # Deterministic "random" coin flip based on current state.
    cur = gripper_hist[-1]
    cell = str(cur.get("cell", "A1"))
    yaw = str(cur.get("yaw", "N"))
    n = int(memory.get("n_interactions", 0))
    score = sum(ord(ch) for ch in (cell + yaw)) + n
    return "translation" if (score % 2 == 0) else "rotation"


def _emit_intent_gate(
    objects: Sequence[Dict],
    gripper_hist: Sequence[Dict],
    memory: Dict,
    state: OracleState,
    *,
    user_state: Optional[UserState],
) -> Optional[Dict]:
    """
    Returns an INTERACT tool call if we can ask a high-signal intent-gating question,
    otherwise returns None.
    """
    current_cell = gripper_hist[-1]["cell"]
    candidates = list(memory.get("candidates", []))
    excluded_obj_ids = set(memory.get("excluded_obj_ids") or [])
    if excluded_obj_ids:
        candidates = [c for c in candidates if c not in excluded_obj_ids]
    mode = _effective_mode(user_state, gripper_hist, memory)

    # For rotation mode, prefer yaw intent gating (if a yaw-oscillation signal exists).
    if mode == "rotation":
        triggered, dom_cell, yaw1, yaw2 = _has_yaw_oscillation(gripper_hist)
        if triggered:
            target_obj = next((o for o in objects if o["cell"] == dom_cell), None)
            if target_obj:
                text = (
                    f"I notice you are struggling aligning the gripper yaw while near the {target_obj['label']}. "
                    f"Is that what you are trying to do?"
                )
                choices = ["1) YES", "2) NO"]
                context = {
                    "type": "intent_gate_yaw",
                    "obj_id": target_obj["id"],
                    "label": target_obj["label"],
                    "action": "ALIGN_YAW",
                }
                return _interact("QUESTION", text, choices, context, state)

    # Candidate-based intent gating (translation or rotation when yaw-signal isn't available).
    ranked = _rank_candidates(objects, candidates, current_cell)
    if len(ranked) >= 2:
        k = min(3, len(ranked))
        a = ranked[0]
        others = ranked[1:k]
        other_labels = ", ".join(o["label"] for o in others)
        if mode == "rotation":
            text = (
                f"I notice you are rotating the gripper near the {a['label']}. However, {other_labels} "
                f"{'is' if len(others)==1 else 'are'} also close. Are you trying to align yaw to one of these?"
            )
            action = "ALIGN_YAW"
        else:
            text = (
                f"I notice you are approaching the {a['label']}. However, {other_labels} "
                f"{'is' if len(others)==1 else 'are'} also close. Are you trying to grasp one of these?"
            )
            action = "APPROACH"
        choices = ["1) YES", "2) NO"]
        context = {
            "type": "intent_gate_candidates",
            "labels": [o["label"] for o in ranked[:k]],
            "action": action,
        }
        return _interact("QUESTION", text, choices, context, state)

    # Otherwise, gate on yaw struggle if present (fallback).
    triggered, dom_cell, yaw1, yaw2 = _has_yaw_oscillation(gripper_hist)
    if triggered:
        target_obj = next((o for o in objects if o["cell"] == dom_cell), None)
        if target_obj:
            text = (
                f"I notice you are struggling aligning the gripper yaw while near the {target_obj['label']}. "
                f"Is that what you are trying to do?"
            )
            choices = ["1) YES", "2) NO"]
            context = {"type": "intent_gate_yaw", "obj_id": target_obj["id"], "label": target_obj["label"], "action": "ALIGN_YAW"}
            return _interact("QUESTION", text, choices, context, state)

    return None


def oracle_decide_tool(
    objects: Sequence[Dict],
    gripper_hist: Sequence[Dict],
    memory: Dict,
    state: OracleState,
    user_state: Optional[UserState] = None,
) -> Dict:
    current_cell = gripper_hist[-1]["cell"]
    candidates = list(memory.get("candidates", []))
    excluded_obj_ids = set(memory.get("excluded_obj_ids") or [])
    if excluded_obj_ids:
        candidates = [c for c in candidates if c not in excluded_obj_ids]
    objects_by_id = {o["id"]: o for o in objects}
    current_yaw = gripper_hist[-1]["yaw"]
    mode = _effective_mode(user_state, gripper_hist, memory)

    if state.terminate_episode:
        # The driver should stop the episode when this is set. Fall back to a single
        # benign prompt to avoid motion tools if called anyway.
        return _interact(
            "SUGGESTION",
            "Okay. I'll stay out of the way.",
            ["1) OK"],
            {"type": "terminal_ack"},
            state,
        )

    # If we just approached an object, optionally offer a follow-up align-yaw assist
    # instead of repeating the last action.
    last_action = memory.get("last_action") or {}
    if (
        isinstance(last_action, dict)
        and last_action.get("tool") == "APPROACH"
        and isinstance(last_action.get("obj"), str)
        and state.pending_action_obj_id is None
        and state.selected_obj_id is None
        and not (state.awaiting_confirmation or state.awaiting_choice or state.awaiting_help or state.awaiting_anything_else or state.awaiting_mode_select or state.awaiting_intent_gate)
    ):
        obj_id = last_action["obj"]
        obj = objects_by_id.get(obj_id)
        if obj and current_cell == obj["cell"] and current_yaw != obj["yaw"]:
            # Ask explicitly before aligning yaw.
            state.selected_obj_id = obj_id
            state.awaiting_confirmation = True
            question = f"Do you want me to also align yaw to the {obj['label']}?"
            context = {"type": "confirm", "obj_id": obj_id, "label": obj["label"], "action": "ALIGN_YAW"}
            choices = ["1) YES", "2) NO"]
            return _interact("CONFIRM", question, choices, context, state)

    # Follow-up action after user confirmed help/approach.
    if state.pending_action_obj_id is not None and state.pending_action_obj_id in objects_by_id:
        target = objects_by_id[state.pending_action_obj_id]
        # A confirmation should trigger exactly ONE motion tool, then return to dialog.
        def clear_pending() -> None:
            state.pending_action_obj_id = None
            state.selected_obj_id = None
            state.pending_mode = None
            state.awaiting_confirmation = False
            state.awaiting_choice = False
            state.awaiting_help = False
            state.awaiting_intent_gate = False
            state.awaiting_anything_else = False
            state.awaiting_mode_select = False

        if state.pending_mode == "APPROACH":
            if current_cell != target["cell"]:
                tool = _tool("APPROACH", {"obj": target["id"]})
                clear_pending()
                return tool
            # Already at target cell; don't re-approach.
            clear_pending()
        elif state.pending_mode == "ALIGN_YAW":
            if current_yaw != target["yaw"]:
                tool = _tool("ALIGN_YAW", {"obj": target["id"]})
                clear_pending()
                return tool
            clear_pending()
        else:
            # No explicit mode; pick a single corrective action.
            if current_cell != target["cell"]:
                tool = _tool("APPROACH", {"obj": target["id"]})
                clear_pending()
                return tool
            if current_yaw != target["yaw"]:
                tool = _tool("ALIGN_YAW", {"obj": target["id"]})
                clear_pending()
                return tool
            clear_pending()

    # If we're waiting for a user reply, keep prompting (don't fall through to motion tools).
    if state.awaiting_confirmation:
        obj_id = state.selected_obj_id or state.intended_obj_id
        obj = objects_by_id.get(obj_id)
        if obj:
            action = state.last_prompt_context.get("action") if state.last_prompt_context else None
            if action == "ALIGN_YAW":
                question = f"Do you want me to align yaw to the {obj['label']}?"
            else:
                question = f"Do you want me to approach the {obj['label']}?"
            choices = ["1) YES", "2) NO"]
            context = {"type": "confirm", "obj_id": obj["id"], "label": obj["label"], "action": action or "APPROACH"}
            return _interact("CONFIRM", question, choices, context, state)
        state.awaiting_confirmation = False

    if state.awaiting_anything_else:
        text = "Uh, I should have misunderstood. Is there anything else I can help with?"
        choices = ["1) YES", "2) NO"]
        context = {"type": "anything_else"}
        return _interact("QUESTION", text, choices, context, state)

    if state.awaiting_mode_select:
        # If the user's current input mode strongly implies the kind of assistance they want,
        # skip the mode-selection prompt and go straight to object selection.
        if mode == "translation":
            state.pending_mode = "APPROACH"
            state.awaiting_mode_select = False
            state.awaiting_choice = True
        elif mode == "rotation":
            state.pending_mode = "ALIGN_YAW"
            state.awaiting_mode_select = False
            state.awaiting_choice = True
        else:
            text = (
                "Do you want help with approaching an object or aligning the gripper yaw to an object?"
            )
            choices = ["1) APPROACH", "2) ALIGN_YAW"]
            context = {"type": "mode_select"}
            return _interact("SUGGESTION", text, choices, context, state)

    if state.awaiting_choice:
        ranked = _rank_candidates(objects, candidates, current_cell)
        if ranked:
            k = min(4, len(ranked))
            labels = [ranked[i]["label"] for i in range(k)]
            obj_ids = [ranked[i]["id"] for i in range(k)]
            choices = [f"{i+1}) {labels[i]}" for i in range(k)]
            # Always include an explicit "none" option to support iterative exclusion.
            none_idx = k + 1
            choices.append(f"{none_idx}) None of them")
            context = {"type": "candidate_choice", "labels": labels, "obj_ids": obj_ids, "none_index": none_idx}
            if state.pending_mode == "ALIGN_YAW":
                prompt = "Which object do you want me to align yaw to?"
            elif state.pending_mode == "APPROACH":
                prompt = "Which object do you want me to help you approach?"
            else:
                prompt = "Uh, which one do you want?"
            return _interact("QUESTION", prompt, choices, context, state)
        state.awaiting_choice = False

    if state.awaiting_help:
        # Re-ask the help prompt if we are still in a yaw-struggle state.
        triggered, dom_cell, yaw1, yaw2 = _has_yaw_oscillation(gripper_hist)
        if triggered:
            target_obj = next((o for o in objects if o["cell"] == dom_cell), None)
            if target_obj and target_obj["yaw"] not in {yaw1, yaw2}:
                text = f"Do you want me to help you align yaw to the {target_obj['label']}?"
                choices = ["1) YES", "2) NO"]
                context = {"type": "help", "obj_id": target_obj["id"], "yaws": (yaw1, yaw2, target_obj["yaw"])}
                return _interact("SUGGESTION", text, choices, context, state)
        state.awaiting_help = False

    if state.awaiting_intent_gate:
        gate = _emit_intent_gate(objects, gripper_hist, memory, state, user_state=user_state)
        if gate is not None:
            return gate
        state.awaiting_intent_gate = False

    # At the very beginning of an episode, force an intent-gating INTERACT before any action.
    # This matches the desired behavior: clarify intent first, then approach/align.
    if int(memory.get("n_interactions", 0)) == 0 and not (memory.get("past_dialogs") or []):
        if not (state.awaiting_confirmation or state.awaiting_choice or state.awaiting_help):
            state.awaiting_intent_gate = True
            gate = _emit_intent_gate(objects, gripper_hist, memory, state, user_state=user_state)
            if gate is not None:
                return gate
            state.awaiting_intent_gate = False

    # Confirmation after a user-picked object.
    if state.selected_obj_id is not None and not state.awaiting_confirmation:
        obj = objects_by_id.get(state.selected_obj_id)
        if obj:
            # Ask to confirm the *next* action we would take (approach vs align yaw),
            # so the post-confirm tool call matches the user's expectation.
            if state.pending_mode == "ALIGN_YAW":
                question = f"Do you want me to align yaw to the {obj['label']}?"
                context = {"type": "confirm", "obj_id": obj["id"], "label": obj["label"], "action": "ALIGN_YAW"}
            elif state.pending_mode == "APPROACH":
                question = f"Do you want me to approach the {obj['label']}?"
                context = {"type": "confirm", "obj_id": obj["id"], "label": obj["label"], "action": "APPROACH"}
            elif current_cell != obj["cell"]:
                question = f"Do you want me to approach the {obj['label']}?"
                context = {"type": "confirm", "obj_id": obj["id"], "label": obj["label"], "action": "APPROACH"}
            elif current_yaw != obj["yaw"]:
                question = f"Do you want me to align yaw to the {obj['label']}?"
                context = {"type": "confirm", "obj_id": obj["id"], "label": obj["label"], "action": "ALIGN_YAW"}
            else:
                # Already at the desired pose; no need to confirm another action.
                state.selected_obj_id = None
                state.awaiting_confirmation = False
                state.awaiting_choice = False
                state.awaiting_help = False
                state.awaiting_intent_gate = False
                state.awaiting_anything_else = False
                state.awaiting_mode_select = False
                obj = None
            if obj is not None:
                state.awaiting_confirmation = True
                choices = ["1) YES", "2) NO"]
                return _interact("CONFIRM", question, choices, context, state)

    # Candidate clarification when approach evidence is similar.
    #
    # Important: don't ask this every timestep. Gate it so it primarily triggers right after
    # a movement step (APPROACH), which makes it feel like the assistant is reacting to
    # ambiguous motion rather than nagging.
    top_two = _top_two_candidates(objects, candidates, current_cell)
    last_calls = list(memory.get("last_tool_calls", []))
    just_moved = bool(last_calls and last_calls[-1] == "APPROACH")
    if top_two and not state.awaiting_choice and not state.awaiting_confirmation:
        a, b = top_two
        osc = _has_cell_oscillation(gripper_hist, a["cell"], b["cell"])
        allow = just_moved or osc
        if allow:
            dist_a = grid.manhattan(current_cell, a["cell"])
            dist_b = grid.manhattan(current_cell, b["cell"])
            if abs(dist_a - dist_b) <= 1:
                # First gate on intent, then ask which specific object.
                state.awaiting_intent_gate = True
                ranked = _rank_candidates(objects, candidates, current_cell)
                k = min(3, len(ranked))
                a0 = ranked[0]
                others = ranked[1:k]
                other_labels = ", ".join(o["label"] for o in others)
                text = (
                    f"I notice you are approaching the {a0['label']}. However, {other_labels} "
                    f"{'is' if len(others)==1 else 'are'} also close. Are you trying to grasp one of these?"
                )
                choices = ["1) YES", "2) NO"]
                context = {"type": "intent_gate_candidates", "labels": [o["label"] for o in ranked[:k]]}
                return _interact("QUESTION", text, choices, context, state)

    # Yaw struggle suggestion.
    triggered, dom_cell, yaw1, yaw2 = _has_yaw_oscillation(gripper_hist)
    if triggered and not state.awaiting_help:
        target_obj = next((o for o in objects if o["cell"] == dom_cell), None)
        if target_obj and target_obj["yaw"] not in {yaw1, yaw2}:
            # First gate on intent: is the user trying to align yaw to this object?
            state.awaiting_intent_gate = True
            text = (
                f"I notice you are struggling aligning the gripper yaw while near the {target_obj['label']}. "
                f"Is that what you are trying to do?"
            )
            choices = ["1) YES", "2) NO"]
            context = {"type": "intent_gate_yaw", "obj_id": target_obj["id"], "label": target_obj["label"]}
            return _interact("QUESTION", text, choices, context, state)

    # Default policy: move toward the intended object or align yaw when co-located.
    intended = objects_by_id[state.intended_obj_id]
    if current_cell != intended["cell"]:
        return _tool("APPROACH", {"obj": intended["id"]})
    if current_yaw != intended["yaw"]:
        return _tool("ALIGN_YAW", {"obj": intended["id"]})

    # If already aligned and close, gently re-confirm intent.
    if not state.awaiting_confirmation:
        state.awaiting_confirmation = True
        text = f"Do you want me to approach the {intended['label']}?"
        choices = ["1) YES", "2) NO"]
        context = {"type": "confirm", "obj_id": intended["id"], "label": intended["label"], "action": "APPROACH"}
        return _interact("CONFIRM", text, choices, context, state)

    # Waiting for a YES/NO; keep prompting.
    text = f"Do you want me to approach the {intended['label']}?"
    choices = ["1) YES", "2) NO"]
    context = {"type": "confirm", "obj_id": intended["id"], "label": intended["label"], "action": "APPROACH"}
    return _interact("CONFIRM", text, choices, context, state)


def validate_tool_call(tool_call: Dict) -> None:
    if not isinstance(tool_call, dict) or set(tool_call.keys()) != {"tool", "args"}:
        raise ValueError("Tool call must be {tool, args}")
    tool = tool_call["tool"]
    args = tool_call["args"]
    if tool not in {"INTERACT", "APPROACH", "ALIGN_YAW"}:
        raise ValueError(f"Invalid tool: {tool}")
    if not isinstance(args, dict):
        raise ValueError("args must be an object")
    if tool == "INTERACT":
        required_keys = {"kind", "text", "choices"}
        if set(args.keys()) != required_keys:
            raise ValueError("INTERACT args must be {kind,text,choices}")
        if args["kind"] not in {"QUESTION", "SUGGESTION", "CONFIRM"}:
            raise ValueError("Invalid INTERACT.kind")
        if not isinstance(args["text"], str):
            raise ValueError("INTERACT.text must be a string")
        choices = args["choices"]
        if not isinstance(choices, list) or not choices or not all(isinstance(c, str) for c in choices):
            raise ValueError("INTERACT.choices must be a non-empty list of strings")
        for c in choices:
            prefix = c.split(")", 1)[0]
            if not prefix.isdigit():
                raise ValueError("INTERACT.choices must start with numbered prefixes like '1)'")
    elif tool in {"APPROACH", "ALIGN_YAW"}:
        if set(args.keys()) != {"obj"} or not isinstance(args["obj"], str):
            raise ValueError(f"{tool} args must be {{obj}}")
