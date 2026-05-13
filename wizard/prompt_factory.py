"""Template-based prompt builder for the wizard GUI.

The wizard's job is **judgement** (when to ask, what to ask about, who the
target is) — not copywriting. This module gives the wizard a fixed bank of
prompt templates that exactly mirror what each oracle would emit for the
same input, so:

* surface text matches the oracle's distribution (no train/eval drift),
* inter-wizard κ measures real disagreement (prompt type + target), not
  surface phrasing,
* wizard-hours drop ~3× vs free-form typing.

Surface here is one function per prompt type per env. Each returns a
fully-validated ``{tool, args}`` dict ready to drop into the JSONL.

A free-text fallback (``build_free_text``) is also provided so wizards can
still author one-off prompts the templates don't cover.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from data_generator import grid as gridlib
from data_generator.oracle import POUR_AMOUNTS


# ----------------------------------------------------------------------------
# Prompt-type catalog (what shows up in the wizard's dropdown, per env)
# ----------------------------------------------------------------------------

# Order matters: shows up top-down in the GUI dropdown. Most-common first.
_PROMPT_TYPES_COMMON = (
    "confirm",
    "candidate_choice",
    "anything_else",
    "mode_select",
    "terminal_ack",
    "other",
)

_PROMPT_TYPES_YCB = (
    "intent_gate_candidates",
    "intent_gate_yaw",
    "help",
) + _PROMPT_TYPES_COMMON

_PROMPT_TYPES_STACKING = (
    "intent_gate_stack",
    "confirm_stack",
    "non_top_redirect",
) + _PROMPT_TYPES_COMMON

_PROMPT_TYPES_POURING = (
    "pitcher_acquisition",
    "intent_gate_pour",
    "confirm_pour",
    "confirm_grab",
    "amount_choice",
    "confirm_amount",
    "cup_full_redirect",
) + _PROMPT_TYPES_COMMON


def list_prompt_types(env: str) -> Tuple[str, ...]:
    if env == "reach_to_grasp_ycb":
        return _PROMPT_TYPES_YCB
    if env == "cube_stacking":
        return _PROMPT_TYPES_STACKING
    if env == "pouring":
        return _PROMPT_TYPES_POURING
    return _PROMPT_TYPES_COMMON


# ----------------------------------------------------------------------------
# What sub-arguments each prompt type needs (drives GUI widget visibility)
# ----------------------------------------------------------------------------

# Each entry is a set of arg-names the wizard must provide.
# Possible names: "target", "action", "amount", "alt_target".
PROMPT_SIGNATURES: Dict[str, Tuple[str, ...]] = {
    # YCB
    "intent_gate_candidates": (),
    "intent_gate_yaw": ("target",),
    "help": ("target",),
    # Stacking
    "intent_gate_stack": (),
    "confirm_stack": ("target",),
    "non_top_redirect": ("target", "alt_target"),
    # Pouring
    "pitcher_acquisition": ("target",),
    "intent_gate_pour": (),
    "confirm_pour": ("target",),
    "confirm_grab": ("target",),
    "amount_choice": ("target",),
    "confirm_amount": ("target", "amount"),
    "cup_full_redirect": ("target", "alt_target"),
    # Common
    "confirm": ("target", "action"),
    "candidate_choice": (),
    "anything_else": (),
    "mode_select": (),
    "terminal_ack": (),
    "other": ("free_text", "free_choices"),
}


def signature(prompt_type: str) -> Tuple[str, ...]:
    return PROMPT_SIGNATURES.get(prompt_type, ())


# Friendly one-liners shown next to the prompt-type name in the dropdown.
# Keep these short — they wrap inside a ttk.Combobox.
PROMPT_LABELS: Dict[str, str] = {
    # YCB
    "intent_gate_candidates": "Are you trying to grasp one of these nearby objects? (Q)",
    "intent_gate_yaw": "Are you trying to align yaw to {target}? (Q)",
    "help": "Help align yaw to {target}? (S)",
    # Stacking
    "intent_gate_stack": "Want me to stack on one of these nearby cubes? (Q)",
    "confirm_stack": "Stack the held cube on {target}? (C)",
    "non_top_redirect": "{target} is covered — stack on {alt_target} instead? (S)",
    # Pouring
    "pitcher_acquisition": "Grab the pitcher for you? (S)",
    "intent_gate_pour": "Want me to pour into one of these nearby cups? (Q)",
    "confirm_pour": "Pour into {target}? (C)",
    "confirm_grab": "Grab the {target}? (C)",
    "amount_choice": "How much to pour into {target}? (Q · SMALL/HALF/FULL)",
    "confirm_amount": "Pour {amount} into {target}? (C)",
    "cup_full_redirect": "{target} is full — pour into {alt_target} instead? (S)",
    # Common
    "confirm": "Confirm next action on {target}: APPROACH/ALIGN_YAW/STACK/GRAB/POUR (C)",
    "candidate_choice": "Which one? — list all viable targets (Q)",
    "anything_else": "Anything else I can help with? — recovery (Q)",
    "mode_select": "Which kind of help? — APPROACH/STACK/POUR/etc. (S)",
    "terminal_ack": "Acknowledgement — staying out of the way (S)",
    "other": "Free-text fallback (custom prompt + choices)",
}


def display_label(prompt_type: str, *, target_label: Optional[str] = None,
                  alt_label: Optional[str] = None, amount: Optional[str] = None) -> str:
    """Render the dropdown text for a prompt type, optionally interpolating context."""
    raw = PROMPT_LABELS.get(prompt_type, prompt_type)
    return f"{prompt_type:24s}  {raw.format(target=target_label or '<target>', alt_target=alt_label or '<alt>', amount=amount or '<amount>')}"


def precondition_status(env: str, prompt_type: str, blob: Dict[str, Any]) -> Tuple[str, str]:
    """Check whether the oracle's decision tree could plausibly emit this prompt
    type given the current state.

    Returns ``(level, message)`` where level ∈ {"ok", "warn", "fail"}:
    * ``ok``   — the oracle could plausibly emit this here.
    * ``warn`` — the oracle wouldn't typically emit this, but the prompt is
                 still schema-valid. (Use sparingly: don't be too clever; let
                 the wizard exercise judgement.)
    * ``fail`` — required state is missing entirely (e.g., ``confirm_amount``
                 picked but no pitcher held). The wizard should pick something else.
    """
    objs = blob.get("objects") or []
    mem = blob.get("memory") or {}

    if env == "cube_stacking":
        held = next((o for o in objs if o.get("is_held")), None)
        non_held_bases = [o for o in objs if not o.get("is_held") and o.get("top_of_stack", True)]
        if prompt_type in {"confirm_stack", "intent_gate_stack", "non_top_redirect"} and held is None:
            return "fail", "no cube is held; stacking-related prompts require a held cube"
        if prompt_type == "non_top_redirect":
            covered = [o for o in objs if not o.get("is_held") and not o.get("top_of_stack", True)]
            if not covered:
                return "warn", "no covered cubes in scene; redirect prompt may not be useful"
        if prompt_type == "intent_gate_stack" and len(non_held_bases) < 2:
            return "warn", "fewer than 2 candidate bases nearby; intent gate degrades to anything_else"

    if env == "pouring":
        pitcher = next((o for o in objs if o.get("kind") == "pitcher"), None)
        pitcher_held = bool(pitcher and pitcher.get("is_held"))
        non_full_cups = [o for o in objs if o.get("kind") == "cup" and o.get("fill") != "FULL"]
        if prompt_type == "pitcher_acquisition" and pitcher_held:
            return "fail", "pitcher is already held; acquisition prompt would be wrong"
        if prompt_type in {"confirm_pour", "amount_choice", "confirm_amount", "cup_full_redirect"} and not pitcher_held:
            return "fail", "pitcher is not held; pour-related prompts require a held pitcher"
        if prompt_type == "intent_gate_pour" and len(non_full_cups) < 2:
            return "warn", "fewer than 2 viable cups nearby; intent gate degrades to anything_else"
        if prompt_type == "cup_full_redirect":
            full_cups = [o for o in objs if o.get("kind") == "cup" and o.get("fill") == "FULL"]
            if not full_cups:
                return "warn", "no full cups in scene; redirect prompt may not be useful"
        if prompt_type == "confirm_grab" and pitcher_held:
            return "warn", "pitcher already held; confirm_grab is unusual here"

    # Generic checks
    if prompt_type == "intent_gate_candidates":
        cands = list(mem.get("candidates") or [])
        if len(cands) < 2:
            return "warn", "fewer than 2 candidates in memory; intent gate degrades"

    return "ok", ""


# ----------------------------------------------------------------------------
# Target filtering — which objects make sense as targets for each prompt
# ----------------------------------------------------------------------------

def valid_targets(env: str, prompt_type: str, blob: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of objects that could legally be the target of this prompt.

    E.g., for ``confirm_stack`` only show top-of-stack non-held cubes; for
    ``confirm_pour`` only show non-full cups; etc.
    """
    objs = list(blob.get("objects") or [])

    if env == "cube_stacking":
        if prompt_type in {"confirm_stack", "non_top_redirect"}:
            return [o for o in objs if not o.get("is_held") and o.get("top_of_stack", True)]

    if env == "pouring":
        if prompt_type in {"confirm_pour", "amount_choice", "confirm_amount", "cup_full_redirect"}:
            return [o for o in objs if o.get("kind") == "cup" and o.get("fill") != "FULL"]
        if prompt_type in {"pitcher_acquisition", "confirm_grab"}:
            return [o for o in objs if o.get("kind") == "pitcher"]

    if prompt_type in {"confirm", "help", "intent_gate_yaw"}:
        return [o for o in objs if not o.get("is_held")]

    return objs


def valid_actions(env: str, prompt_type: str) -> Tuple[str, ...]:
    """Return the action choices the wizard can attach to a ``confirm`` prompt
    (i.e., what motion tool the YES reply would unlock)."""
    if env == "reach_to_grasp_ycb":
        return ("APPROACH", "ALIGN_YAW")
    if env == "cube_stacking":
        return ("APPROACH", "ALIGN_YAW", "STACK")
    if env == "pouring":
        return ("APPROACH", "ALIGN_YAW", "GRAB", "POUR")
    return ("APPROACH",)


def valid_amounts() -> Tuple[str, ...]:
    return POUR_AMOUNTS


# ----------------------------------------------------------------------------
# Internal helpers (mirroring data_generator/oracle.py)
# ----------------------------------------------------------------------------

def _objects_by_id(blob: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {o["id"]: o for o in (blob.get("objects") or [])}


def _ranked_by_distance(blob: Dict[str, Any], pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cur_cell = blob["gripper_hist"][-1]["cell"]
    return sorted(pool, key=lambda o: (gridlib.manhattan(cur_cell, o["cell"]), o["id"]))


def _held(blob: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for o in blob.get("objects") or []:
        if o.get("is_held"):
            return o
    return None


def _interact(kind: str, text: str, choices: List[str], context: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build the ``{tool, args}`` dict + companion context that mimics oracle._interact."""
    return (
        {"tool": "INTERACT", "args": {"kind": kind, "text": text, "choices": list(choices)}},
        context,
    )


# ----------------------------------------------------------------------------
# Builders — one per prompt type, mirroring the oracle templates 1:1
# ----------------------------------------------------------------------------

def _build_intent_gate_candidates(env: str, blob: Dict[str, Any], *, action: str = "APPROACH") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    candidates = list(blob.get("memory", {}).get("candidates") or [])
    excluded = set(blob.get("memory", {}).get("excluded_obj_ids") or [])
    candidates = [c for c in candidates if c not in excluded]
    pool = [o for o in (blob.get("objects") or []) if o.get("id") in set(candidates) and not o.get("is_held")]
    ranked = _ranked_by_distance(blob, pool)
    if len(ranked) < 2:
        # Degenerate; fall back to anything_else.
        return _build_anything_else()
    k = min(3, len(ranked))
    a0 = ranked[0]
    others = ranked[1:k]
    other_labels = ", ".join(o["label"] for o in others)
    if action == "ALIGN_YAW":
        text = (
            f"I notice you are rotating the gripper near the {a0['label']}. However, {other_labels} "
            f"{'is' if len(others)==1 else 'are'} also close. Are you trying to align yaw to one of these?"
        )
    else:
        text = (
            f"I notice you are approaching the {a0['label']}. However, {other_labels} "
            f"{'is' if len(others)==1 else 'are'} also close. Are you trying to grasp one of these?"
        )
    return _interact(
        "QUESTION", text, ["1) YES", "2) NO"],
        {"type": "intent_gate_candidates", "labels": [o["label"] for o in ranked[:k]], "action": action},
    )


def _build_intent_gate_yaw(blob: Dict[str, Any], *, target_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    obj = _objects_by_id(blob).get(target_id, {})
    text = (
        f"I notice you are struggling aligning the gripper yaw while near the {obj.get('label', target_id)}. "
        "Is that what you are trying to do?"
    )
    return _interact(
        "QUESTION", text, ["1) YES", "2) NO"],
        {"type": "intent_gate_yaw", "obj_id": target_id, "label": obj.get("label", ""), "action": "ALIGN_YAW"},
    )


def _build_intent_gate_stack(blob: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    candidates = list(blob.get("memory", {}).get("candidates") or [])
    excluded = set(blob.get("memory", {}).get("excluded_obj_ids") or [])
    candidates = [c for c in candidates if c not in excluded]
    pool = [o for o in (blob.get("objects") or []) if o.get("id") in set(candidates) and not o.get("is_held")]
    ranked = _ranked_by_distance(blob, pool)
    held = _held(blob)
    held_label = held["label"] if held else "the held cube"
    if len(ranked) < 2:
        return _build_anything_else()
    k = min(3, len(ranked))
    a0 = ranked[0]
    others = ranked[1:k]
    other_labels = ", ".join(o["label"] for o in others)
    text = (
        f"I notice you're moving {held_label} near the {a0['label']}. "
        f"{other_labels} {'is' if len(others)==1 else 'are'} also close. "
        "Want me to stack on one of these?"
    )
    return _interact(
        "QUESTION", text, ["1) YES", "2) NO"],
        {"type": "intent_gate_stack", "labels": [o["label"] for o in ranked[:k]], "action": "STACK"},
    )


def _build_intent_gate_pour(blob: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    candidates = list(blob.get("memory", {}).get("candidates") or [])
    excluded = set(blob.get("memory", {}).get("excluded_obj_ids") or [])
    candidates = [c for c in candidates if c not in excluded]
    pool = [o for o in (blob.get("objects") or []) if o.get("id") in set(candidates) and o.get("kind") == "cup"]
    ranked = _ranked_by_distance(blob, pool)
    if len(ranked) < 2:
        return _build_anything_else()
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
        "QUESTION", text, ["1) YES", "2) NO"],
        {"type": "intent_gate_pour", "labels": [o["label"] for o in ranked[:k]], "action": "POUR"},
    )


def _build_confirm(blob: Dict[str, Any], *, target_id: str, action: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    obj = _objects_by_id(blob).get(target_id, {})
    label = obj.get("label", target_id)
    action = (action or "APPROACH").upper()
    if action == "ALIGN_YAW":
        text = f"Do you want me to align yaw to the {label}?"
        ptype = "confirm"
    elif action == "STACK":
        held = _held(blob)
        hl = held["label"] if held else "the held cube"
        text = f"Do you want me to stack the {hl} on the {label}?"
        ptype = "confirm_stack"
    elif action == "GRAB":
        text = f"Do you want me to grab the {label}?"
        ptype = "confirm_grab"
    elif action == "POUR":
        text = f"Do you want me to pour into the {label}?"
        ptype = "confirm_pour"
    else:
        text = f"Do you want me to approach the {label}?"
        ptype = "confirm"
    return _interact(
        "CONFIRM", text, ["1) YES", "2) NO"],
        {"type": ptype, "obj_id": target_id, "label": label, "action": action},
    )


def _build_confirm_stack(blob: Dict[str, Any], *, target_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return _build_confirm(blob, target_id=target_id, action="STACK")


def _build_confirm_pour(blob: Dict[str, Any], *, target_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return _build_confirm(blob, target_id=target_id, action="POUR")


def _build_confirm_grab(blob: Dict[str, Any], *, target_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return _build_confirm(blob, target_id=target_id, action="GRAB")


def _build_pitcher_acquisition(blob: Dict[str, Any], *, target_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    obj = _objects_by_id(blob).get(target_id, {})
    text = f"I see the {obj.get('label', 'pitcher')} on the table. Should I grab it for you?"
    return _interact(
        "SUGGESTION", text, ["1) YES", "2) NO"],
        {"type": "pitcher_acquisition", "obj_id": target_id, "label": obj.get("label", "pitcher"), "action": "GRAB"},
    )


def _build_non_top_redirect(blob: Dict[str, Any], *, target_id: str, alt_target_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    objs = _objects_by_id(blob)
    covered = objs.get(target_id, {}).get("label", target_id)
    alt = objs.get(alt_target_id, {}).get("label", alt_target_id)
    text = f"The {covered} already has a cube on top. Want to stack on the {alt} instead?"
    return _interact(
        "SUGGESTION", text, ["1) YES", "2) NO"],
        {"type": "non_top_redirect", "alt_obj_id": alt_target_id, "alt_label": alt},
    )


def _build_cup_full_redirect(blob: Dict[str, Any], *, target_id: str, alt_target_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    objs = _objects_by_id(blob)
    full = objs.get(target_id, {}).get("label", target_id)
    alt = objs.get(alt_target_id, {}).get("label", alt_target_id)
    text = f"The {full} is already full. Pour into the {alt} instead?"
    return _interact(
        "SUGGESTION", text, ["1) YES", "2) NO"],
        {"type": "cup_full_redirect", "alt_obj_id": alt_target_id, "alt_label": alt},
    )


def _build_amount_choice(blob: Dict[str, Any], *, target_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    obj = _objects_by_id(blob).get(target_id, {})
    label = obj.get("label", target_id)
    amounts = list(POUR_AMOUNTS)
    choices = [f"{i+1}) {a}" for i, a in enumerate(amounts)]
    none_idx = len(amounts) + 1
    choices.append(f"{none_idx}) None — don't pour")
    return _interact(
        "QUESTION", f"How much should I pour into the {label}?", choices,
        {"type": "amount_choice", "obj_id": target_id, "amounts": amounts, "none_index": none_idx},
    )


def _build_confirm_amount(blob: Dict[str, Any], *, target_id: str, amount: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    obj = _objects_by_id(blob).get(target_id, {})
    label = obj.get("label", target_id)
    amount = amount.upper()
    return _interact(
        "CONFIRM", f"Pour {amount} into the {label}?", ["1) YES", "2) NO"],
        {"type": "confirm_amount", "obj_id": target_id, "amount": amount, "label": label},
    )


def _build_help(blob: Dict[str, Any], *, target_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    obj = _objects_by_id(blob).get(target_id, {})
    label = obj.get("label", target_id)
    return _interact(
        "SUGGESTION", f"Do you want me to help you align yaw to the {label}?", ["1) YES", "2) NO"],
        {"type": "help", "obj_id": target_id, "label": label, "action": "ALIGN_YAW"},
    )


def _build_candidate_choice(env: str, blob: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build a 'Which one?' prompt listing all currently-viable targets."""
    objs = blob.get("objects") or []
    excluded = set(blob.get("memory", {}).get("excluded_obj_ids") or [])

    if env == "cube_stacking":
        pool = [o for o in objs if not o.get("is_held") and o.get("top_of_stack", True) and o.get("id") not in excluded]
        prompt = "Which cube should I stack on?"
    elif env == "pouring":
        pool = [o for o in objs if o.get("kind") == "cup" and o.get("fill") != "FULL" and o.get("id") not in excluded]
        prompt = "Which cup should I pour into?"
    else:  # ycb
        pool = [o for o in objs if not o.get("is_held") and o.get("id") not in excluded]
        prompt = "Which one do you want?"

    pool = _ranked_by_distance(blob, pool)
    if not pool:
        return _build_anything_else()
    k = min(4, len(pool))
    labels = [pool[i]["label"] for i in range(k)]
    obj_ids = [pool[i]["id"] for i in range(k)]
    choices = [f"{i+1}) {labels[i]}" for i in range(k)]
    none_idx = k + 1
    choices.append(f"{none_idx}) None of them")
    return _interact(
        "QUESTION", prompt, choices,
        {"type": "candidate_choice", "labels": labels, "obj_ids": obj_ids, "none_index": none_idx},
    )


def _build_anything_else() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return _interact(
        "QUESTION", "Uh, I should have misunderstood. Is there anything else I can help with?",
        ["1) YES", "2) NO"], {"type": "anything_else"},
    )


def _build_terminal_ack() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return _interact(
        "SUGGESTION", "Okay. I'll stay out of the way.", ["1) OK"], {"type": "terminal_ack"},
    )


def _build_mode_select(env: str, blob: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if env == "cube_stacking":
        actions = ["APPROACH", "STACK"]
        text = "Do you want help approaching the base cube or stacking the held cube?"
    elif env == "pouring":
        pitcher = next((o for o in (blob.get("objects") or []) if o.get("kind") == "pitcher"), None)
        if pitcher and not pitcher.get("is_held"):
            actions = ["GRAB", "APPROACH"]
            text = "Should I grab the pitcher or approach a cup first?"
        else:
            actions = ["APPROACH", "POUR"]
            text = "Do you want help approaching a cup or pouring now?"
    else:
        actions = ["APPROACH", "ALIGN_YAW"]
        text = "Do you want help with approaching an object or aligning the gripper yaw to an object?"
    choices = [f"{i+1}) {a}" for i, a in enumerate(actions)]
    return _interact(
        "SUGGESTION", text, choices, {"type": "mode_select", "actions": actions},
    )


def _build_free_text(*, free_text: str, free_choices: List[str], kind: str = "QUESTION") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Fallback for prompts the templates don't cover."""
    cleaned: List[str] = []
    for i, c in enumerate(free_choices):
        s = c.strip()
        if not s:
            continue
        if not s.split(")", 1)[0].isdigit():
            s = f"{i+1}) {s}"
        cleaned.append(s)
    return (
        {"tool": "INTERACT", "args": {"kind": kind, "text": free_text, "choices": cleaned}},
        {"type": "other"},
    )


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------

def build_prompt(
    env: str,
    prompt_type: str,
    blob: Dict[str, Any],
    *,
    target_id: Optional[str] = None,
    alt_target_id: Optional[str] = None,
    action: Optional[str] = None,
    amount: Optional[str] = None,
    free_text: Optional[str] = None,
    free_choices: Optional[List[str]] = None,
    free_kind: str = "QUESTION",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return ``(tool_call, prompt_context)`` for the given prompt_type and args.

    ``prompt_context`` mirrors what the oracle stores in ``state.last_prompt_context``
    and is used by the GUI's user-reply simulator to know which template fired.
    Raises ``ValueError`` if required kwargs are missing.
    """
    pt = prompt_type
    needed = signature(pt)

    if "target" in needed and target_id is None:
        raise ValueError(f"prompt_type={pt!r} requires target_id")
    if "alt_target" in needed and alt_target_id is None:
        raise ValueError(f"prompt_type={pt!r} requires alt_target_id")
    if "action" in needed and not action:
        raise ValueError(f"prompt_type={pt!r} requires action")
    if "amount" in needed and not amount:
        raise ValueError(f"prompt_type={pt!r} requires amount")
    if pt == "other" and (free_text is None or free_choices is None):
        raise ValueError("prompt_type='other' requires free_text and free_choices")

    if pt == "intent_gate_candidates":
        return _build_intent_gate_candidates(env, blob, action=(action or "APPROACH"))
    if pt == "intent_gate_yaw":
        return _build_intent_gate_yaw(blob, target_id=target_id)
    if pt == "intent_gate_stack":
        return _build_intent_gate_stack(blob)
    if pt == "intent_gate_pour":
        return _build_intent_gate_pour(blob)
    if pt == "confirm":
        return _build_confirm(blob, target_id=target_id, action=action)
    if pt == "confirm_stack":
        return _build_confirm_stack(blob, target_id=target_id)
    if pt == "confirm_pour":
        return _build_confirm_pour(blob, target_id=target_id)
    if pt == "confirm_grab":
        return _build_confirm_grab(blob, target_id=target_id)
    if pt == "pitcher_acquisition":
        return _build_pitcher_acquisition(blob, target_id=target_id)
    if pt == "non_top_redirect":
        return _build_non_top_redirect(blob, target_id=target_id, alt_target_id=alt_target_id)
    if pt == "cup_full_redirect":
        return _build_cup_full_redirect(blob, target_id=target_id, alt_target_id=alt_target_id)
    if pt == "amount_choice":
        return _build_amount_choice(blob, target_id=target_id)
    if pt == "confirm_amount":
        return _build_confirm_amount(blob, target_id=target_id, amount=amount)
    if pt == "help":
        return _build_help(blob, target_id=target_id)
    if pt == "candidate_choice":
        return _build_candidate_choice(env, blob)
    if pt == "anything_else":
        return _build_anything_else()
    if pt == "terminal_ack":
        return _build_terminal_ack()
    if pt == "mode_select":
        return _build_mode_select(env, blob)
    if pt == "other":
        return _build_free_text(free_text=free_text, free_choices=free_choices, kind=free_kind)
    raise ValueError(f"Unknown prompt_type: {pt!r}")
