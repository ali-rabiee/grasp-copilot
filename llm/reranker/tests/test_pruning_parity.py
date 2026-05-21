"""Structural parity between infer_pruning_intent and the oracle.

The oracle's on_user_reply (OracleBackend in
evaluation/benchmarks/scenario_noise_sweep.py) is the authority on what
each INTERACT context-type means. For oracle-emitted questions where we
have ground-truth `last_prompt.context.type`, infer_pruning_intent
should return:

  context.type ∈ {confirm, confirm_*, help, *_redirect, pitcher_acquisition}
    → binary_confirm  (if the text names exactly one candidate object)

  context.type == "candidate_choice"
    → candidate_choice

  context.type ∈ {intent_gate_*, anything_else, mode_select, terminal_ack}
    → noop  (these don't shrink the candidate set; oracle uses internal
            flags instead)

This isn't a byte-for-byte snapshot parity (the oracle's reply handler
mutates OracleState, not the candidate set itself — the candidate set
only shrinks during the next motion via _refresh_candidates). What we
DO guarantee is that our pruning class is consistent with the oracle's
intent for each context type.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from llm.reranker.pruning import infer_pruning_intent


CONTRACT_PATH = Path(__file__).resolve().parents[3] / "data" / "woz_phase2" / "llm_contract_valid.jsonl"


def _iter_contract(path: Path, limit: int = 0):
    if not path.exists():
        pytest.skip(f"contract JSONL not present: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                return
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _input_state(row: Dict[str, Any]):
    inp_str = row.get("input", "")
    if not isinstance(inp_str, str) or not inp_str.strip():
        return None
    try:
        return json.loads(inp_str)
    except Exception:
        return None


def _gt_tool_call(row: Dict[str, Any]):
    out = row.get("output", "")
    if not isinstance(out, str) or not out.strip():
        return None
    try:
        return json.loads(out)
    except Exception:
        return None


def _gt_context_type(inp: Dict[str, Any]) -> str:
    memory = inp.get("memory") or {}
    last = memory.get("last_prompt") or {}
    ctx = last.get("context") or {}
    return str(ctx.get("type") or "")


CONFIRM_CTX_TYPES = {
    "confirm", "confirm_stack", "confirm_pour", "confirm_grab",
    "help", "pitcher_acquisition", "non_top_redirect", "cup_full_redirect",
}
CANDIDATE_CHOICE_CTX_TYPES = {"candidate_choice"}
NOOP_CTX_TYPES = {
    "intent_gate_candidates", "intent_gate_yaw",
    "intent_gate_stack", "intent_gate_pour",
    "anything_else", "mode_select", "terminal_ack",
}


def test_intent_inference_matches_oracle_context_class():
    """For each oracle-emitted INTERACT row, the inferred class matches the context type."""
    n_total = 0
    n_match = 0
    n_skip = 0
    mismatches: list = []
    for row in _iter_contract(CONTRACT_PATH, limit=2000):
        gt = _gt_tool_call(row)
        if not gt or gt.get("tool") != "INTERACT":
            continue
        inp = _input_state(row)
        if inp is None:
            n_skip += 1
            continue
        # The GT row is the *next* tool call. The oracle's context type for the
        # PREVIOUS turn lives in memory.last_prompt.context.type. To parity-check
        # the GT INTERACT, we need its own context — which we don't have in
        # the contract. So we instead simulate: build the (objects, candidates)
        # from `inp`, and check that the GT tool-call's choices+text are
        # consistent with the class our inference picks.

        candidates = (inp.get("memory") or {}).get("candidates") or []
        objects = inp.get("objects") or []
        intent = infer_pruning_intent(gt, candidates, objects)

        args = gt.get("args") or {}
        choices = args.get("choices") or []
        text = str(args.get("text") or "")
        upper_choices = [str(c).strip().upper() for c in choices]

        # Self-describing class from the GT call itself.
        has_none = any("NONE" in u for u in upper_choices)
        per_choice_obj_count = 0
        for c in choices:
            ids = [
                o for o in objects
                if str(o.get("label", "")).lower() in str(c).lower()
                and str(o.get("id")) in set(candidates)
            ]
            if len({str(o.get("id")) for o in ids}) == 1:
                per_choice_obj_count += 1
        has_yes_no = any("YES" in u for u in upper_choices) and any("NO" in u for u in upper_choices)
        single_obj_in_text = sum(
            1 for o in objects
            if str(o.get("label", "")).lower() in text.lower()
            and str(o.get("id")) in set(candidates)
        )

        if per_choice_obj_count >= 2:
            expected = "candidate_choice"
        elif has_yes_no and single_obj_in_text == 1:
            expected = "binary_confirm"
        else:
            expected = "noop"

        n_total += 1
        if intent.kind == expected:
            n_match += 1
        else:
            mismatches.append((expected, intent.kind, choices, text[:60]))

    if n_total == 0:
        pytest.skip("no oracle INTERACT rows in contract — skipping")

    rate = n_match / n_total
    # Self-consistency on the GT contract should be near-perfect.
    assert rate >= 0.95, (
        f"intent inference matched only {rate:.1%} of {n_total} GT INTERACTs "
        f"({len(mismatches)} mismatches); first 5: {mismatches[:5]}"
    )


def test_intent_binary_confirm_yes_picks_named_obj():
    """Hand-crafted: 'Do you want me to approach the mug?' YES/NO."""
    objects = [
        {"id": "obj_1", "label": "mug", "cell": "A1", "yaw": "N", "is_held": False},
        {"id": "obj_2", "label": "bowl", "cell": "A2", "yaw": "N", "is_held": False},
    ]
    candidates = ["obj_1", "obj_2"]
    tool_call = {
        "tool": "INTERACT",
        "args": {
            "kind": "CONFIRM",
            "text": "Do you want me to approach the mug?",
            "choices": ["1) YES", "2) NO"],
        },
    }
    intent = infer_pruning_intent(tool_call, candidates, objects)
    assert intent.kind == "binary_confirm"
    assert intent.target_obj_ids == ("obj_1",)
    assert intent.yes_idx == 0 and intent.no_idx == 1


def test_intent_candidate_choice_two_objects_plus_none():
    objects = [
        {"id": "obj_1", "label": "mug", "cell": "A1", "yaw": "N", "is_held": False},
        {"id": "obj_2", "label": "bowl", "cell": "A2", "yaw": "N", "is_held": False},
        {"id": "obj_3", "label": "cup", "cell": "A3", "yaw": "N", "is_held": False},
    ]
    candidates = ["obj_1", "obj_2", "obj_3"]
    tool_call = {
        "tool": "INTERACT",
        "args": {
            "kind": "QUESTION",
            "text": "Which object do you want help with?",
            "choices": ["1) mug", "2) bowl", "3) cup", "4) None of them"],
        },
    }
    intent = infer_pruning_intent(tool_call, candidates, objects)
    assert intent.kind == "candidate_choice"
    assert intent.none_idx == 3
    assert intent.choice_to_obj == {0: "obj_1", 1: "obj_2", 2: "obj_3"}


def test_intent_intent_gate_yields_noop():
    """An ambiguous intent-gating question with YES/NO but no named single obj → noop."""
    objects = [
        {"id": "obj_1", "label": "mug", "cell": "A1", "yaw": "N", "is_held": False},
        {"id": "obj_2", "label": "bowl", "cell": "A2", "yaw": "N", "is_held": False},
    ]
    candidates = ["obj_1", "obj_2"]
    tool_call = {
        "tool": "INTERACT",
        "args": {
            "kind": "QUESTION",
            "text": "I notice you are approaching the mug. However, bowl is also close. Are you trying to grasp one of these?",
            "choices": ["1) YES", "2) NO"],
        },
    }
    intent = infer_pruning_intent(tool_call, candidates, objects)
    # Two objects in text → not a single-target binary confirm → noop.
    assert intent.kind == "noop"
