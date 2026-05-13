"""Pouring oracle with three-step "what / where / how much" sub-flow.

Decision tree adds five new prompt types on top of the YCB invariants:

* ``pitcher_acquisition`` — SUGGESTION: "grab the pitcher first?"
* ``intent_gate_pour``     — QUESTION: gate intent before picking a cup.
* ``confirm_pour``         — CONFIRM: "pour into cup_A?"
* ``amount_choice``        — QUESTION: 3 buckets {SMALL, HALF, FULL} + None.
* ``confirm_amount``       — CONFIRM: "pour HALF into cup_A?"

Plus ``cup_full_redirect`` (SUGGESTION) when the chosen target is already FULL.

The amount sub-flow runs ONLY after target+pitcher are confirmed. It uses two
new OracleState flags (``awaiting_amount_choice``, ``awaiting_amount_confirmation``)
and ``state.pending_amount`` to carry the wizard's pick into the final POUR call.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from . import grid
from . import yaw as yawlib
from .oracle import (
    POUR_AMOUNTS,
    OracleState,
    UserState,
    _effective_mode,
    _has_cell_oscillation,
    _has_yaw_oscillation,
    _interact,
    _rank_candidates,
    _tool,
    _top_two_candidates,
)


def _pitcher(objects: Sequence[Dict]) -> Optional[Dict]:
    return next((o for o in objects if o.get("kind") == "pitcher"), None)


def _cups(objects: Sequence[Dict]) -> List[Dict]:
    return [o for o in objects if o.get("kind") == "cup"]


def _pourable_cups(objects: Sequence[Dict]) -> List[Dict]:
    """Cups that are not already FULL — valid pour targets."""
    return [c for c in _cups(objects) if c.get("fill") != "FULL"]


def _emit_intent_gate_pour(
    objects: Sequence[Dict],
    gripper_hist: Sequence[Dict],
    memory: Dict,
    state: OracleState,
) -> Optional[Dict]:
    current_cell = gripper_hist[-1]["cell"]
    candidates = list(memory.get("candidates") or [])
    excluded = set(memory.get("excluded_obj_ids") or [])
    candidates = [c for c in candidates if c not in excluded]
    ranked = _rank_candidates(objects, candidates, current_cell)
    if len(ranked) >= 2:
        k = min(3, len(ranked))
        a0 = ranked[0]
        others = ranked[1:k]
        other_labels = ", ".join(o["label"] for o in others)
        text = (
            f"I notice you're heading toward the {a0['label']}. "
            f"{other_labels} {'is' if len(others)==1 else 'are'} also close. "
            "Want me to pour into one of these?"
        )
        return _interact(
            "QUESTION",
            text,
            ["1) YES", "2) NO"],
            {
                "type": "intent_gate_pour",
                "labels": [o["label"] for o in ranked[:k]],
                "action": "POUR",
            },
            state,
        )
    return None


def pouring_decide_tool(
    objects: Sequence[Dict],
    gripper_hist: Sequence[Dict],
    memory: Dict,
    state: OracleState,
    user_state: Optional[UserState] = None,
) -> Dict:
    current_cell = gripper_hist[-1]["cell"]
    current_yaw = gripper_hist[-1]["yaw"]
    candidates = list(memory.get("candidates") or [])
    excluded = set(memory.get("excluded_obj_ids") or [])
    candidates = [c for c in candidates if c not in excluded]
    objects_by_id = {o["id"]: o for o in objects}
    pitcher = _pitcher(objects)
    pitcher_held = bool(pitcher and pitcher.get("is_held"))

    # ---------------------------------------------------------- terminal guard
    if state.terminate_episode:
        return _interact(
            "SUGGESTION",
            "Okay. I'll stay out of the way.",
            ["1) OK"],
            {"type": "terminal_ack"},
            state,
        )

    # -------------------- post-APPROACH follow-up: at target cup, ask CONFIRM pour
    last_action = memory.get("last_action") or {}
    if (
        isinstance(last_action, dict)
        and last_action.get("tool") == "APPROACH"
        and isinstance(last_action.get("obj"), str)
        and pitcher_held
        and state.pending_action_obj_id is None
        and state.selected_obj_id is None
        and not (
            state.awaiting_confirmation or state.awaiting_choice or state.awaiting_help
            or state.awaiting_anything_else or state.awaiting_mode_select
            or state.awaiting_intent_gate or state.awaiting_amount_choice
            or state.awaiting_amount_confirmation
        )
    ):
        obj_id = last_action["obj"]
        cup = objects_by_id.get(obj_id)
        if cup and cup.get("kind") == "cup" and current_cell == cup["cell"]:
            state.selected_obj_id = obj_id
            state.awaiting_confirmation = True
            state.pending_mode = "POUR"
            text = f"Do you want me to pour into the {cup['label']}?"
            return _interact(
                "CONFIRM",
                text,
                ["1) YES", "2) NO"],
                {"type": "confirm_pour", "obj_id": obj_id, "label": cup["label"], "action": "POUR"},
                state,
            )

    # ------------------- amount sub-flow re-prompts (these dominate motion tools)
    if state.awaiting_amount_confirmation:
        cup = objects_by_id.get(state.selected_obj_id) if state.selected_obj_id else None
        amount = state.pending_amount or "HALF"
        cup_label = cup["label"] if cup else "the cup"
        return _interact(
            "CONFIRM",
            f"Pour {amount} into the {cup_label}?",
            ["1) YES", "2) NO"],
            {"type": "confirm_amount", "obj_id": state.selected_obj_id, "amount": amount, "label": cup_label},
            state,
        )

    if state.awaiting_amount_choice:
        cup = objects_by_id.get(state.selected_obj_id) if state.selected_obj_id else None
        cup_label = cup["label"] if cup else "the cup"
        choices = [f"{i+1}) {amt}" for i, amt in enumerate(POUR_AMOUNTS)]
        none_idx = len(POUR_AMOUNTS) + 1
        choices.append(f"{none_idx}) None — don't pour")
        return _interact(
            "QUESTION",
            f"How much should I pour into the {cup_label}?",
            choices,
            {
                "type": "amount_choice",
                "obj_id": state.selected_obj_id,
                "amounts": list(POUR_AMOUNTS),
                "none_index": none_idx,
            },
            state,
        )

    # ----------------------------------------- pending action confirmed by user
    if state.pending_action_obj_id is not None and state.pending_action_obj_id in objects_by_id:
        target = objects_by_id[state.pending_action_obj_id]

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
            clear_pending()
        elif state.pending_mode == "ALIGN_YAW":
            if current_yaw != target["yaw"]:
                tool = _tool("ALIGN_YAW", {"obj": target["id"]})
                clear_pending()
                return tool
            clear_pending()
        elif state.pending_mode == "GRAB":
            if target.get("kind") == "pitcher" and not target.get("is_held"):
                if current_cell != target["cell"]:
                    tool = _tool("APPROACH", {"obj": target["id"]})
                    # don't clear pending: still need to GRAB after we arrive.
                    return tool
                tool = _tool("GRAB", {"obj": target["id"]})
                clear_pending()
                return tool
            clear_pending()
        elif state.pending_mode == "POUR":
            # POUR needs an amount. If we don't have one yet, route into the sub-flow
            # *without* clearing the pending action — we'll fire POUR after confirmation.
            if state.pending_amount is None:
                state.awaiting_amount_choice = True
                # Keep selected_obj_id set; clear the pending_action flag so we don't loop here.
                state.pending_action_obj_id = None
                return pouring_decide_tool(objects, gripper_hist, memory, state, user_state=user_state)
            tool = _tool("POUR", {"obj": target["id"], "amount": state.pending_amount})
            state.pending_amount = None
            clear_pending()
            return tool

    # ---------------------------------------------------------- awaiting cascade
    if state.awaiting_confirmation:
        obj_id = state.selected_obj_id or state.intended_obj_id
        obj = objects_by_id.get(obj_id)
        if obj:
            action = state.last_prompt_context.get("action") if state.last_prompt_context else None
            if action == "ALIGN_YAW":
                text = f"Do you want me to align yaw to the {obj['label']}?"
                ptype = "confirm"
            elif action == "GRAB":
                text = f"Do you want me to grab the {obj['label']}?"
                ptype = "confirm_grab"
            elif action == "POUR":
                text = f"Do you want me to pour into the {obj['label']}?"
                ptype = "confirm_pour"
            else:
                text = f"Do you want me to approach the {obj['label']}?"
                ptype = "confirm"
            return _interact(
                "CONFIRM",
                text,
                ["1) YES", "2) NO"],
                {"type": ptype, "obj_id": obj["id"], "label": obj["label"], "action": action or "APPROACH"},
                state,
            )
        state.awaiting_confirmation = False

    if state.awaiting_anything_else:
        return _interact(
            "QUESTION",
            "Uh, I should have misunderstood. Is there anything else I can help with?",
            ["1) YES", "2) NO"],
            {"type": "anything_else"},
            state,
        )

    if state.awaiting_mode_select:
        if not pitcher_held:
            return _interact(
                "SUGGESTION",
                "Should I grab the pitcher or approach a cup first?",
                ["1) GRAB", "2) APPROACH"],
                {"type": "mode_select", "actions": ["GRAB", "APPROACH"]},
                state,
            )
        return _interact(
            "SUGGESTION",
            "Do you want help approaching a cup or pouring now?",
            ["1) APPROACH", "2) POUR"],
            {"type": "mode_select", "actions": ["APPROACH", "POUR"]},
            state,
        )

    if state.awaiting_choice:
        # Once the user is in the help-flow, show *all* viable cups, not only
        # those within the gripper's intent-radius. Otherwise the only-cup-across-
        # the-grid case lands the user in a "none of those" → anything-else loop.
        excluded = set(memory.get("excluded_obj_ids") or [])
        all_cups = [
            o for o in objects
            if o.get("kind") == "cup" and o.get("fill") != "FULL" and o.get("id") not in excluded
        ]
        all_cups.sort(key=lambda o: (grid.manhattan(current_cell, o["cell"]), o["id"]))
        if all_cups:
            k = min(4, len(all_cups))
            labels = [all_cups[i]["label"] for i in range(k)]
            obj_ids = [all_cups[i]["id"] for i in range(k)]
            choices = [f"{i+1}) {labels[i]}" for i in range(k)]
            none_idx = k + 1
            choices.append(f"{none_idx}) None of them")
            return _interact(
                "QUESTION",
                "Which cup should I pour into?",
                choices,
                {"type": "candidate_choice", "labels": labels, "obj_ids": obj_ids, "none_index": none_idx},
                state,
            )
        state.awaiting_choice = False
        state.awaiting_anything_else = True
        return _interact(
            "QUESTION",
            "Okay — none of those. Is there anything else I can help with?",
            ["1) YES", "2) NO"],
            {"type": "anything_else"},
            state,
        )

    if state.awaiting_intent_gate:
        gate = _emit_intent_gate_pour(objects, gripper_hist, memory, state)
        if gate is not None:
            return gate
        state.awaiting_intent_gate = False

    # -------------------------------- force episode-start intent gate
    if int(memory.get("n_interactions", 0)) == 0 and not (memory.get("past_dialogs") or []):
        if not pitcher_held and pitcher is not None:
            state.awaiting_confirmation = True
            state.selected_obj_id = pitcher["id"]
            state.pending_mode = "GRAB"
            text = f"I see the pitcher on the table. Should I grab it for you?"
            return _interact(
                "SUGGESTION",
                text,
                ["1) YES", "2) NO"],
                {"type": "pitcher_acquisition", "obj_id": pitcher["id"], "label": pitcher["label"], "action": "GRAB"},
                state,
            )
        state.awaiting_intent_gate = True
        gate = _emit_intent_gate_pour(objects, gripper_hist, memory, state)
        if gate is not None:
            return gate
        state.awaiting_intent_gate = False

    # --------------------------- pitcher acquisition (any time pitcher not held)
    if not pitcher_held and pitcher is not None and not (
        state.awaiting_confirmation or state.awaiting_choice
        or state.awaiting_amount_choice or state.awaiting_amount_confirmation
    ):
        state.awaiting_confirmation = True
        state.selected_obj_id = pitcher["id"]
        state.pending_mode = "GRAB"
        text = f"I should grab the {pitcher['label']} first — OK?"
        return _interact(
            "SUGGESTION",
            text,
            ["1) YES", "2) NO"],
            {"type": "pitcher_acquisition", "obj_id": pitcher["id"], "label": pitcher["label"], "action": "GRAB"},
            state,
        )

    # ----------------------------------------- cup-full redirect when applicable
    intended = objects_by_id.get(state.intended_obj_id)
    if intended is not None and intended.get("kind") == "cup" and intended.get("fill") == "FULL":
        pourable = _pourable_cups(objects)
        if pourable:
            alt = sorted(pourable, key=lambda c: grid.manhattan(current_cell, c["cell"]))[0]
            state.awaiting_intent_gate = True
            return _interact(
                "SUGGESTION",
                f"The {intended['label']} is already full. Pour into the {alt['label']} instead?",
                ["1) YES", "2) NO"],
                {"type": "cup_full_redirect", "alt_obj_id": alt["id"], "alt_label": alt["label"]},
                state,
            )

    # ------------------- selected-obj confirmation (after user picked something)
    # Must run BEFORE candidate clarification to avoid an oscillating history
    # repeatedly re-emitting the intent gate after the user already picked a cup.
    if state.selected_obj_id is not None and not state.awaiting_confirmation:
        obj = objects_by_id.get(state.selected_obj_id)
        if obj is not None:
            if obj.get("kind") == "pitcher" and not obj.get("is_held"):
                action = "GRAB"
                text = f"Do you want me to grab the {obj['label']}?"
                ptype = "confirm_grab"
            elif state.pending_mode == "ALIGN_YAW":
                action = "ALIGN_YAW"
                text = f"Do you want me to align yaw to the {obj['label']}?"
                ptype = "confirm"
            elif state.pending_mode == "APPROACH" or current_cell != obj["cell"]:
                action = "APPROACH"
                text = f"Do you want me to approach the {obj['label']}?"
                ptype = "confirm"
            else:
                action = "POUR"
                text = f"Do you want me to pour into the {obj['label']}?"
                ptype = "confirm_pour"
            state.awaiting_confirmation = True
            state.pending_mode = action
            return _interact(
                "CONFIRM",
                text,
                ["1) YES", "2) NO"],
                {"type": ptype, "obj_id": obj["id"], "label": obj["label"], "action": action},
                state,
            )

    # --------------- candidate clarification when two cups similarly close
    # Extra ``selected_obj_id is None`` guard so we don't re-prompt for
    # disambiguation after the user has already picked a cup.
    top_two = _top_two_candidates(objects, candidates, current_cell)
    last_calls = list(memory.get("last_tool_calls") or [])
    just_moved = bool(last_calls and last_calls[-1] == "APPROACH")
    if (
        top_two
        and state.selected_obj_id is None
        and state.pending_action_obj_id is None
        and not state.awaiting_choice
        and not state.awaiting_confirmation
    ):
        a, b = top_two
        osc = _has_cell_oscillation(gripper_hist, a["cell"], b["cell"])
        if just_moved or osc:
            dist_a = grid.manhattan(current_cell, a["cell"])
            dist_b = grid.manhattan(current_cell, b["cell"])
            if abs(dist_a - dist_b) <= 1:
                state.awaiting_intent_gate = True
                gate = _emit_intent_gate_pour(objects, gripper_hist, memory, state)
                if gate is not None:
                    return gate
                state.awaiting_intent_gate = False

    # ------------------------------------------------------------ default policy
    if pitcher is None:
        state.awaiting_anything_else = True
        return _interact(
            "QUESTION",
            "I don't see a pitcher. Anything else I can help with?",
            ["1) YES", "2) NO"],
            {"type": "anything_else"},
            state,
        )

    if not pitcher_held:
        # Always CONFIRM before any motion toward the pitcher.
        state.awaiting_confirmation = True
        state.selected_obj_id = pitcher["id"]
        if current_cell != pitcher["cell"]:
            state.pending_mode = "APPROACH"
            return _interact(
                "CONFIRM",
                f"Do you want me to approach the {pitcher['label']}?",
                ["1) YES", "2) NO"],
                {"type": "confirm", "obj_id": pitcher["id"], "label": pitcher["label"], "action": "APPROACH"},
                state,
            )
        state.pending_mode = "GRAB"
        return _interact(
            "CONFIRM",
            f"Do you want me to grab the {pitcher['label']}?",
            ["1) YES", "2) NO"],
            {"type": "confirm_grab", "obj_id": pitcher["id"], "label": pitcher["label"], "action": "GRAB"},
            state,
        )

    if intended is None:
        state.awaiting_anything_else = True
        return _interact(
            "QUESTION",
            "I don't have a clear pour target. Anything else?",
            ["1) YES", "2) NO"],
            {"type": "anything_else"},
            state,
        )

    # ALWAYS CONFIRM, never silent motion.
    if current_cell != intended["cell"]:
        state.awaiting_confirmation = True
        state.selected_obj_id = intended["id"]
        state.pending_mode = "APPROACH"
        return _interact(
            "CONFIRM",
            f"Do you want me to approach the {intended['label']}?",
            ["1) YES", "2) NO"],
            {"type": "confirm", "obj_id": intended["id"], "label": intended["label"], "action": "APPROACH"},
            state,
        )
    if current_yaw != intended["yaw"]:
        state.awaiting_confirmation = True
        state.selected_obj_id = intended["id"]
        state.pending_mode = "ALIGN_YAW"
        return _interact(
            "CONFIRM",
            f"Do you want me to align yaw to the {intended['label']}?",
            ["1) YES", "2) NO"],
            {"type": "confirm", "obj_id": intended["id"], "label": intended["label"], "action": "ALIGN_YAW"},
            state,
        )

    # At the target cup, pitcher held → CONFIRM pour, then amount sub-flow.
    state.awaiting_confirmation = True
    state.selected_obj_id = intended["id"]
    state.pending_mode = "POUR"
    return _interact(
        "CONFIRM",
        f"Do you want me to pour into the {intended['label']}?",
        ["1) YES", "2) NO"],
        {"type": "confirm_pour", "obj_id": intended["id"], "label": intended["label"], "action": "POUR"},
        state,
    )
