"""Cube-stacking oracle.

Decision tree (see ``wizard/README.md`` § 5 for the planning doc) — mirrors
the YCB oracle's structure invariant for invariant, with three new branches:

* ``intent_gate_stack``     — early disambiguation among nearby base cubes.
* ``non_top_redirect``      — suggest a different base if the chosen one is covered.
* ``confirm_stack``         — final CONFIRM before the STACK motion call.

Shared helpers come from ``oracle.py`` so the awaiting cascade, the
``_rank_candidates`` ranking, oscillation detectors, and the validator all stay
in lock-step with YCB.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from . import grid
from . import yaw as yawlib
from .oracle import (
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


def _stack_bases(objects: Sequence[Dict]) -> List[Dict]:
    """Plausible stack targets: not held, top of stack."""
    return [o for o in objects if not o.get("is_held") and o.get("top_of_stack", True)]


def _held_obj(objects: Sequence[Dict]) -> Optional[Dict]:
    return next((o for o in objects if o.get("is_held")), None)


def _emit_intent_gate_stack(
    objects: Sequence[Dict],
    gripper_hist: Sequence[Dict],
    memory: Dict,
    state: OracleState,
    *,
    user_state: Optional[UserState],
) -> Optional[Dict]:
    """High-signal intent gate for stacking, before asking 'which base?'."""
    current_cell = gripper_hist[-1]["cell"]
    candidates = list(memory.get("candidates") or [])
    excluded = set(memory.get("excluded_obj_ids") or [])
    candidates = [c for c in candidates if c not in excluded]

    ranked = _rank_candidates(objects, candidates, current_cell)
    held = _held_obj(objects)
    held_label = held["label"] if held else "the held cube"

    if len(ranked) >= 2:
        k = min(3, len(ranked))
        a0 = ranked[0]
        others = ranked[1:k]
        other_labels = ", ".join(o["label"] for o in others)
        text = (
            f"I notice you're moving {held_label} near the {a0['label']}. "
            f"{other_labels} {'is' if len(others)==1 else 'are'} also close. "
            "Want me to stack on one of these?"
        )
        choices = ["1) YES", "2) NO"]
        context = {
            "type": "intent_gate_stack",
            "labels": [o["label"] for o in ranked[:k]],
            "action": "STACK",
        }
        return _interact("QUESTION", text, choices, context, state)
    return None


def stacking_decide_tool(
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
    held = _held_obj(objects)

    # ---------------------------------------------------- terminal / fallback
    if state.terminate_episode:
        return _interact(
            "SUGGESTION",
            "Okay. I'll stay out of the way.",
            ["1) OK"],
            {"type": "terminal_ack"},
            state,
        )

    # If nothing is held, the stacking task is over (success or RELEASE).
    # There's no recovery from this state — you can't stack without holding a
    # cube — so terminate the episode cleanly instead of routing to recovery
    # prompts that would loop.
    if held is None:
        state.terminate_episode = True
        return _interact(
            "SUGGESTION",
            "Okay — nothing held. I'll stay out of the way.",
            ["1) OK"],
            {"type": "terminal_ack"},
            state,
        )

    # -------------------------- post-APPROACH follow-up: ask CONFIRM "stack here?"
    last_action = memory.get("last_action") or {}
    if (
        isinstance(last_action, dict)
        and last_action.get("tool") == "APPROACH"
        and isinstance(last_action.get("obj"), str)
        and held is not None
        and state.pending_action_obj_id is None
        and state.selected_obj_id is None
        and not (
            state.awaiting_confirmation or state.awaiting_choice or state.awaiting_help
            or state.awaiting_anything_else or state.awaiting_mode_select
            or state.awaiting_intent_gate
        )
    ):
        base_id = last_action["obj"]
        base = objects_by_id.get(base_id)
        if base and current_cell == base["cell"]:
            state.selected_obj_id = base_id
            state.awaiting_confirmation = True
            state.pending_mode = "STACK"
            text = f"Do you want me to stack the {held['label']} on the {base['label']}?"
            choices = ["1) YES", "2) NO"]
            context = {"type": "confirm_stack", "obj_id": base_id, "label": base["label"], "action": "STACK"}
            return _interact("CONFIRM", text, choices, context, state)

    # ------------------------------------------ pending action confirmed by user
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
        elif state.pending_mode == "STACK":
            if held is not None and current_cell == target["cell"]:
                tool = _tool("STACK", {"obj": target["id"]})
                clear_pending()
                return tool
            # Not yet at the base — approach first.
            if current_cell != target["cell"]:
                tool = _tool("APPROACH", {"obj": target["id"]})
                clear_pending()
                return tool
            clear_pending()
        else:
            if current_cell != target["cell"]:
                tool = _tool("APPROACH", {"obj": target["id"]})
                clear_pending()
                return tool
            if current_yaw != target["yaw"]:
                tool = _tool("ALIGN_YAW", {"obj": target["id"]})
                clear_pending()
                return tool
            clear_pending()

    # ---------------------------------------------------------- awaiting cascade
    if state.awaiting_confirmation:
        obj_id = state.selected_obj_id or state.intended_obj_id
        obj = objects_by_id.get(obj_id)
        if obj:
            action = state.last_prompt_context.get("action") if state.last_prompt_context else None
            if action == "ALIGN_YAW":
                text = f"Do you want me to align yaw to the {obj['label']}?"
                ptype = "confirm"
            elif action == "STACK":
                hl = held["label"] if held else "the held cube"
                text = f"Do you want me to stack the {hl} on the {obj['label']}?"
                ptype = "confirm_stack"
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
        return _interact(
            "SUGGESTION",
            "Do you want help approaching the base cube or stacking the held cube?",
            ["1) APPROACH", "2) STACK"],
            {"type": "mode_select", "actions": ["APPROACH", "STACK"]},
            state,
        )

    if state.awaiting_choice:
        # Same as pouring: once in the help-flow, show every uncovered base
        # (ignoring candidate_max_dist) so the user can always pick a target.
        excluded = set(memory.get("excluded_obj_ids") or [])
        all_bases = [
            o for o in objects
            if not o.get("is_held") and o.get("top_of_stack", True) and o.get("id") not in excluded
        ]
        all_bases.sort(key=lambda o: (grid.manhattan(current_cell, o["cell"]), o["id"]))
        if all_bases:
            k = min(4, len(all_bases))
            labels = [all_bases[i]["label"] for i in range(k)]
            obj_ids = [all_bases[i]["id"] for i in range(k)]
            choices = [f"{i+1}) {labels[i]}" for i in range(k)]
            none_idx = k + 1
            choices.append(f"{none_idx}) None of them")
            prompt = (
                "Which cube should I stack on?"
                if state.pending_mode == "STACK" or held is not None
                else "Which one do you want?"
            )
            return _interact(
                "QUESTION",
                prompt,
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
        gate = _emit_intent_gate_stack(objects, gripper_hist, memory, state, user_state=user_state)
        if gate is not None:
            return gate
        state.awaiting_intent_gate = False

    # ----------------------------------------- force episode-start intent gate
    if int(memory.get("n_interactions", 0)) == 0 and not (memory.get("past_dialogs") or []):
        state.awaiting_intent_gate = True
        gate = _emit_intent_gate_stack(objects, gripper_hist, memory, state, user_state=user_state)
        if gate is not None:
            return gate
        state.awaiting_intent_gate = False

    # --------------------------- non-top redirect: chosen base already covered
    intended = objects_by_id.get(state.intended_obj_id)
    if (
        intended is not None
        and not intended.get("top_of_stack", True)
        and not state.awaiting_confirmation
    ):
        # Pick the closest uncovered base to suggest instead.
        bases = _stack_bases(objects)
        if bases:
            ranked = sorted(bases, key=lambda o: grid.manhattan(current_cell, o["cell"]))
            alt = ranked[0]
            state.awaiting_intent_gate = True
            text = (
                f"The {intended['label']} already has a cube on top. Want to stack on "
                f"the {alt['label']} instead?"
            )
            return _interact(
                "SUGGESTION",
                text,
                ["1) YES", "2) NO"],
                {"type": "non_top_redirect", "alt_obj_id": alt["id"], "alt_label": alt["label"]},
                state,
            )

    # ------------------- selected-obj confirmation (after user picked something)
    # Must run BEFORE candidate clarification, otherwise an oscillating gripper
    # history will keep re-emitting the intent gate even after the user has
    # already picked a base — causing a candidate_choice ↔ intent_gate loop.
    if state.selected_obj_id is not None and not state.awaiting_confirmation:
        obj = objects_by_id.get(state.selected_obj_id)
        if obj is not None:
            if state.pending_mode == "ALIGN_YAW":
                text = f"Do you want me to align yaw to the {obj['label']}?"
                ptype, action = "confirm", "ALIGN_YAW"
            elif state.pending_mode == "STACK":
                hl = held["label"] if held else "the held cube"
                text = f"Do you want me to stack the {hl} on the {obj['label']}?"
                ptype, action = "confirm_stack", "STACK"
            elif state.pending_mode == "APPROACH":
                text = f"Do you want me to approach the {obj['label']}?"
                ptype, action = "confirm", "APPROACH"
            elif current_cell != obj["cell"]:
                text = f"Do you want me to approach the {obj['label']}?"
                ptype, action = "confirm", "APPROACH"
            elif held is not None:
                hl = held["label"]
                text = f"Do you want me to stack the {hl} on the {obj['label']}?"
                ptype, action = "confirm_stack", "STACK"
            else:
                state.selected_obj_id = None
                obj = None
            if obj is not None:
                state.awaiting_confirmation = True
                return _interact(
                    "CONFIRM",
                    text,
                    ["1) YES", "2) NO"],
                    {"type": ptype, "obj_id": obj["id"], "label": obj["label"], "action": action},
                    state,
                )

    # ----------------------- candidate clarification when two bases similar dist
    # Extra ``selected_obj_id is None`` guard: once a base is selected, don't
    # re-prompt for ambiguity disambiguation.
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
                gate = _emit_intent_gate_stack(objects, gripper_hist, memory, state, user_state=user_state)
                if gate is not None:
                    return gate
                state.awaiting_intent_gate = False

    # ----------------------------------- default policy: ALWAYS CONFIRM, never silent motion
    if intended is None:
        bases = _stack_bases(objects)
        if bases:
            state.intended_obj_id = bases[0]["id"]
            intended = bases[0]
        else:
            state.awaiting_anything_else = True
            return _interact(
                "QUESTION",
                "All bases are covered. Anything else I can help with?",
                ["1) YES", "2) NO"],
                {"type": "anything_else"},
                state,
            )

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

    # At the base, aligned, holding something — CONFIRM stack.
    state.awaiting_confirmation = True
    state.selected_obj_id = intended["id"]
    state.pending_mode = "STACK"
    hl = held["label"] if held else "the held cube"
    return _interact(
        "CONFIRM",
        f"Do you want me to stack the {hl} on the {intended['label']}?",
        ["1) YES", "2) NO"],
        {"type": "confirm_stack", "obj_id": intended["id"], "label": intended["label"], "action": "STACK"},
        state,
    )
