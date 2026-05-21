"""Entropy + information-gain math for the IG reranker.

  IG(question) = H(C_before) - E_r[H(C | r)]

where C is the candidate set, r is the user's reply, and H is Shannon
entropy in bits. The expectation is taken under a prior P(r) which is
either uniform (default; matches the brief) or motion-weighted (re-uses
the SA1 baseline's inverse-distance softmax).
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from llm.reranker.pruning import PruneIntent, PruneSnapshot, simulate_reply


def entropy_bits(items: Sequence[str], *, prior: Optional[Mapping[str, float]] = None) -> float:
    """Shannon entropy in bits over `items`.

    With prior=None, items are uniform-weighted (empty → 0, singleton → 0).
    With an explicit prior, items missing from the prior get weight 0 (we
    normalise over the intersection).
    """
    n = len(items)
    if n <= 1:
        return 0.0
    if prior is None:
        # log2(n)
        return math.log2(n)
    weights = [max(float(prior.get(i, 0.0)), 0.0) for i in items]
    z = sum(weights)
    if z <= 0:
        return math.log2(n)
    h = 0.0
    for w in weights:
        if w <= 0:
            continue
        p = w / z
        h -= p * math.log2(p)
    return h


def motion_weighted_prior(
    candidate_obj_ids: Sequence[str],
    gripper_hist: Sequence[Dict],
    objects: Sequence[Dict],
    *,
    beta: float = 1.5,
    dir_w: float = 0.5,
) -> Dict[str, float]:
    """Inverse-distance softmax weighted by recent gripper-motion direction.

    Same shape as the SA1 baseline in
    evaluation/benchmarks/offline_exec_benchmark.py:_heuristic_predict_then_assist.
    Used as an optional prior over candidate objects when picking the
    candidate-choice prior; falls back to uniform if `gripper_hist` is empty.
    """
    if not gripper_hist or not candidate_obj_ids:
        return {oid: 1.0 / max(len(candidate_obj_ids), 1) for oid in candidate_obj_ids}
    try:
        from data_generator.grid import manhattan as _grid_manhattan
    except Exception:
        return {oid: 1.0 / len(candidate_obj_ids) for oid in candidate_obj_ids}

    objs_by_id = {str(o.get("id")): o for o in objects}
    cur_cell = gripper_hist[-1].get("cell", "")
    prev_cell = gripper_hist[-2].get("cell", "") if len(gripper_hist) >= 2 else cur_cell

    scores: Dict[str, float] = {}
    for oid in candidate_obj_ids:
        o = objs_by_id.get(oid)
        if not o:
            continue
        ocell = o.get("cell", "")
        d_cur = _grid_manhattan(cur_cell, ocell) if cur_cell and ocell else 0
        score = -beta * d_cur
        if prev_cell and prev_cell != cur_cell:
            d_prev = _grid_manhattan(prev_cell, ocell)
            score += dir_w * (d_prev - d_cur)
        scores[oid] = score

    if not scores:
        return {oid: 1.0 / len(candidate_obj_ids) for oid in candidate_obj_ids}
    m = max(scores.values())
    exps = {k: math.exp(v - m) for k, v in scores.items()}
    z = sum(exps.values()) or 1.0
    return {k: v / z for k, v in exps.items()}


# ── reply priors ───────────────────────────────────────────────────────────


def _uniform_reply_prior(n_choices: int) -> List[float]:
    if n_choices <= 0:
        return []
    return [1.0 / n_choices] * n_choices


def _motion_reply_prior(
    intent: PruneIntent,
    n_choices: int,
    state_before: PruneSnapshot,
    gripper_hist: Sequence[Dict],
    objects: Sequence[Dict],
) -> List[float]:
    """Reweight YES/NO and per-object choices using the motion prior.

    For binary_confirm: P(YES) = candidate prior mass on the named object;
    P(NO) = 1 - P(YES). Other slots get 0.
    For candidate_choice: P(pick i) ∝ prior(obj_at_choice_i); none-of-them
    gets the remaining mass (1 - sum of listed obj priors).
    Else: uniform.
    """
    if n_choices <= 0:
        return []
    if intent.kind == "noop":
        return _uniform_reply_prior(n_choices)

    prior = motion_weighted_prior(state_before.candidates, gripper_hist, objects)
    p = [0.0] * n_choices

    if intent.kind == "binary_confirm":
        obj_id = intent.target_obj_ids[0] if intent.target_obj_ids else None
        p_yes = float(prior.get(obj_id, 0.0)) if obj_id else 0.5
        p_yes = max(min(p_yes, 1.0), 0.0)
        if intent.yes_idx is not None and 0 <= intent.yes_idx < n_choices:
            p[intent.yes_idx] = p_yes
        if intent.no_idx is not None and 0 <= intent.no_idx < n_choices:
            p[intent.no_idx] = 1.0 - p_yes
        z = sum(p) or 1.0
        return [v / z for v in p]

    if intent.kind == "candidate_choice":
        listed_mass = 0.0
        for i, oid in intent.choice_to_obj.items():
            if 0 <= i < n_choices:
                w = float(prior.get(oid, 0.0))
                p[i] = w
                listed_mass += w
        if intent.none_idx is not None and 0 <= intent.none_idx < n_choices:
            p[intent.none_idx] = max(1.0 - listed_mass, 0.0)
        z = sum(p) or 1.0
        return [v / z for v in p]

    return _uniform_reply_prior(n_choices)


# ── core scoring ───────────────────────────────────────────────────────────


def expected_post_entropy(
    intent: PruneIntent,
    n_choices: int,
    state_before: PruneSnapshot,
    *,
    prior: str = "uniform",
    gripper_hist: Sequence[Dict] = (),
    objects: Sequence[Dict] = (),
    candidate_prior: Optional[Mapping[str, float]] = None,
) -> Tuple[float, List[Tuple[int, float, float]]]:
    """Return (E_r[H(C|r)], per-reply [(reply_idx, P(r), H_after)] ).

    `candidate_prior` is the prior used for H(C) itself (uniform if None).
    `prior` is the prior used over user replies — one of {"uniform", "motion_weighted"}.
    """
    if n_choices <= 0:
        return entropy_bits(state_before.candidates, prior=candidate_prior), []

    if prior == "motion_weighted":
        rp = _motion_reply_prior(intent, n_choices, state_before, gripper_hist, objects)
    else:
        rp = _uniform_reply_prior(n_choices)

    e_post = 0.0
    per_reply: List[Tuple[int, float, float]] = []
    for i in range(n_choices):
        p_r = rp[i] if i < len(rp) else 0.0
        after = simulate_reply(state_before, intent, i)
        h_after = entropy_bits(after.candidates, prior=candidate_prior)
        per_reply.append((i, p_r, h_after))
        e_post += p_r * h_after
    return e_post, per_reply


def information_gain_bits(
    intent: PruneIntent,
    n_choices: int,
    state_before: PruneSnapshot,
    *,
    prior: str = "uniform",
    gripper_hist: Sequence[Dict] = (),
    objects: Sequence[Dict] = (),
    candidate_prior: Optional[Mapping[str, float]] = None,
) -> Tuple[float, float, float, List[Tuple[int, float, float]]]:
    """Return (ig_bits, h_before, h_after_expected, per_reply)."""
    h_before = entropy_bits(state_before.candidates, prior=candidate_prior)
    h_after, per_reply = expected_post_entropy(
        intent, n_choices, state_before,
        prior=prior, gripper_hist=gripper_hist, objects=objects,
        candidate_prior=candidate_prior,
    )
    ig = max(h_before - h_after, 0.0)
    return ig, h_before, h_after, per_reply
