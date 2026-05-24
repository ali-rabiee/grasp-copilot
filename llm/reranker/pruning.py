"""Candidate-set pruning for IG scoring.

The oracle's reply handler (OracleBackend.on_user_reply in
evaluation/benchmarks/scenario_noise_sweep.py) updates the candidate set
based on the user's reply and the question's `last_prompt_context.type`.
LLM-generated INTERACT calls do NOT carry that context — they only carry
{kind, text, choices}. This module infers the pruning intent from those
three fields, then applies the same candidate-set delta the oracle would.

Two intent classes cover ~all measurably-informative INTERACT calls:

  * binary_confirm(obj_id): a YES/NO question naming a single candidate.
      YES -> candidates := [obj_id]
      NO  -> candidates := candidates - [obj_id]

  * candidate_choice(obj_ids, has_none): a menu of objects + optional
      "None of them" escape.
      pick obj_i -> candidates := [obj_i]
      pick None  -> candidates := candidates - obj_ids

Anything else (intent_gate, mode_select, anything_else, generic
suggestions) leaves the candidate set unchanged — IG over that set is 0.

The parity test in tests/test_pruning_parity.py verifies that for every
*oracle-emitted* INTERACT in the WoZ valid set, infer_pruning_intent +
simulate_reply produce the same candidate delta as the oracle's reply
handler. New LLM-generated questions never hit a different code path —
they share the same inference + pruning rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# ── snapshot ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PruneSnapshot:
    """Minimal state needed to compute candidate-set entropy."""
    candidates: Tuple[str, ...]          # obj_ids
    excluded_obj_ids: Tuple[str, ...]    # what's been excluded so far

    @classmethod
    def from_memory(cls, memory: Dict) -> "PruneSnapshot":
        cands = tuple(memory.get("candidates") or [])
        excl = tuple(memory.get("excluded_obj_ids") or [])
        # Drop already-excluded from the live candidate list.
        excl_set = set(excl)
        live = tuple(c for c in cands if c not in excl_set)
        return cls(candidates=live, excluded_obj_ids=excl)


# ── intent inference ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class PruneIntent:
    """What `simulate_reply` will do for each reply index."""
    kind: str                            # "binary_confirm" | "candidate_choice" | "noop"
    target_obj_ids: Tuple[str, ...] = ()  # obj_id (binary_confirm) or all listed (candidate_choice)
    yes_idx: Optional[int] = None        # binary_confirm only — which choice means YES
    no_idx: Optional[int] = None
    choice_to_obj: Dict[int, str] = field(default_factory=dict)  # candidate_choice only: reply_idx -> obj_id
    none_idx: Optional[int] = None       # candidate_choice only: which choice means "None of them"


_YES_TOKENS = {"YES", "OK", "OKAY", "CONFIRM", "SURE", "YEAH", "YEP"}
_NO_TOKENS = {"NO", "NOPE", "NAH"}
_NONE_TOKENS = ("NONE OF THEM", "NONE OF THE ABOVE", "NONE")


def _norm(s: str) -> str:
    """Strip numbering and punctuation, uppercase."""
    s = re.sub(r"^\s*\d+\s*[\)\.\:\-]\s*", "", s or "")
    return s.strip().upper()


def _is_yes(choice_norm: str) -> bool:
    toks = set(re.findall(r"[A-Z]+", choice_norm))
    return bool(toks & _YES_TOKENS) and not (toks & _NO_TOKENS)


def _is_no(choice_norm: str) -> bool:
    toks = set(re.findall(r"[A-Z]+", choice_norm))
    return bool(toks & _NO_TOKENS) and not (toks & _YES_TOKENS)


def _is_none_of_them(choice_norm: str) -> bool:
    return any(tok in choice_norm for tok in _NONE_TOKENS)


def _label_to_ids(label: str, candidates: Sequence[str], objects: Sequence[Dict]) -> List[str]:
    """All candidate obj_ids whose label (case-insensitive substring) matches.

    Why substring: the oracle's prompt text says 'the mug' for label 'mug',
    and the LLM may say 'mug_1' or 'small mug'. Substring on label_lower in
    label_text_lower is the most forgiving match that still resolves to a
    real candidate id.
    """
    if not label:
        return []
    label_norm = label.strip().lower()
    if not label_norm:
        return []
    out: List[str] = []
    cand_set = set(candidates)
    for o in objects:
        oid = str(o.get("id", ""))
        if oid not in cand_set:
            continue
        olabel = str(o.get("label", "")).strip().lower()
        if not olabel:
            continue
        if olabel in label_norm or label_norm in olabel:
            out.append(oid)
    return out


def _find_object_in_text(
    text: str, candidates: Sequence[str], objects: Sequence[Dict]
) -> List[str]:
    """Candidate obj_ids whose label appears in the (lowercased) text."""
    if not text:
        return []
    tl = text.lower()
    cand_set = set(candidates)
    out: List[str] = []
    for o in objects:
        oid = str(o.get("id", ""))
        if oid not in cand_set:
            continue
        olabel = str(o.get("label", "")).strip().lower()
        if olabel and olabel in tl:
            out.append(oid)
    return out


def infer_pruning_intent(
    tool_call: Dict,
    candidates: Sequence[str],
    objects: Sequence[Dict],
) -> PruneIntent:
    """Infer which pruning class applies to this INTERACT call.

    The classes match the candidate-set deltas the oracle applies for
    confirm / candidate_choice contexts. intent_gate / anything_else /
    mode_select map to noop here because they don't change the candidate
    set — their information value lies in flags the oracle uses to pick
    the *next* question, which the LLM-side reranker can't model.
    """
    if not isinstance(tool_call, dict) or tool_call.get("tool") != "INTERACT":
        return PruneIntent(kind="noop")
    args = tool_call.get("args") or {}
    text = str(args.get("text") or "")
    _raw_choices = args.get("choices")
    choices = list(_raw_choices) if isinstance(_raw_choices, (list, tuple)) else []
    if not choices:
        return PruneIntent(kind="noop")

    norms = [_norm(c) for c in choices]

    # candidate_choice: a numbered menu where most choices map to a candidate
    # label. Detect by counting how many choices resolve to a single
    # candidate id via their own text (not via the prompt text).
    per_choice_objs: Dict[int, str] = {}
    none_idx: Optional[int] = None
    for i, n in enumerate(norms):
        if _is_none_of_them(n):
            none_idx = i
            continue
        ids = _label_to_ids(n, candidates, objects)
        if len(ids) == 1:
            per_choice_objs[i] = ids[0]

    if len(per_choice_objs) >= 2:
        all_listed = tuple(sorted(set(per_choice_objs.values())))
        return PruneIntent(
            kind="candidate_choice",
            target_obj_ids=all_listed,
            choice_to_obj=per_choice_objs,
            none_idx=none_idx,
        )

    # binary_confirm: YES/NO (in any order) and exactly one candidate object
    # is named in the prompt text.
    yes_idx = next((i for i, n in enumerate(norms) if _is_yes(n)), None)
    no_idx = next((i for i, n in enumerate(norms) if _is_no(n)), None)
    if yes_idx is not None and no_idx is not None:
        objs_in_text = _find_object_in_text(text, candidates, objects)
        if len(objs_in_text) == 1:
            return PruneIntent(
                kind="binary_confirm",
                target_obj_ids=(objs_in_text[0],),
                yes_idx=yes_idx,
                no_idx=no_idx,
            )

    return PruneIntent(kind="noop")


# ── pruning ────────────────────────────────────────────────────────────────


def simulate_reply(
    state_before: PruneSnapshot,
    intent: PruneIntent,
    reply_idx: int,
) -> PruneSnapshot:
    """Apply the inferred pruning rule for one reply index. Returns a new snapshot."""
    cands = list(state_before.candidates)
    excl = list(state_before.excluded_obj_ids)

    if intent.kind == "binary_confirm":
        obj_id = intent.target_obj_ids[0] if intent.target_obj_ids else None
        if obj_id is None:
            return state_before
        if reply_idx == intent.yes_idx:
            new_cands = tuple([obj_id]) if obj_id in cands else state_before.candidates
            return PruneSnapshot(candidates=new_cands, excluded_obj_ids=state_before.excluded_obj_ids)
        if reply_idx == intent.no_idx:
            new_cands = tuple(c for c in cands if c != obj_id)
            if obj_id not in excl:
                excl = sorted(set(excl) | {obj_id})
            return PruneSnapshot(candidates=new_cands, excluded_obj_ids=tuple(excl))
        return state_before

    if intent.kind == "candidate_choice":
        if reply_idx == intent.none_idx:
            listed = set(intent.target_obj_ids)
            new_cands = tuple(c for c in cands if c not in listed)
            new_excl = tuple(sorted(set(excl) | listed))
            return PruneSnapshot(candidates=new_cands, excluded_obj_ids=new_excl)
        picked = intent.choice_to_obj.get(reply_idx)
        if picked is None:
            return state_before
        new_cands = tuple([picked]) if picked in cands else state_before.candidates
        return PruneSnapshot(candidates=new_cands, excluded_obj_ids=state_before.excluded_obj_ids)

    return state_before
