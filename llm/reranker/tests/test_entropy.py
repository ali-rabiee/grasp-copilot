"""Closed-form entropy + IG sanity checks."""

from __future__ import annotations

import math

import pytest

from llm.reranker.entropy import (
    entropy_bits,
    expected_post_entropy,
    information_gain_bits,
)
from llm.reranker.pruning import PruneIntent, PruneSnapshot


def test_entropy_uniform_two_is_one_bit():
    assert entropy_bits(["a", "b"]) == pytest.approx(1.0)


def test_entropy_uniform_four_is_two_bits():
    assert entropy_bits(["a", "b", "c", "d"]) == pytest.approx(2.0)


def test_entropy_singleton_is_zero():
    assert entropy_bits(["a"]) == 0.0
    assert entropy_bits([]) == 0.0


def test_entropy_with_prior_skewed():
    # P=[0.9, 0.1] → H ≈ 0.469 bits
    h = entropy_bits(["a", "b"], prior={"a": 0.9, "b": 0.1})
    expected = -(0.9 * math.log2(0.9) + 0.1 * math.log2(0.1))
    assert h == pytest.approx(expected, rel=1e-6)


def test_binary_confirm_collapses_to_one_bit_at_p05():
    # 2 candidates {x, y}; binary_confirm on x with uniform reply prior.
    # YES → {x} (H=0); NO → {y} (H=0). Expected post = 0. IG = log2(2) = 1.
    snap = PruneSnapshot(candidates=("x", "y"), excluded_obj_ids=())
    intent = PruneIntent(
        kind="binary_confirm",
        target_obj_ids=("x",),
        yes_idx=0,
        no_idx=1,
    )
    ig, h_b, h_a, per_r = information_gain_bits(intent, n_choices=2, state_before=snap)
    assert h_b == pytest.approx(1.0)
    assert h_a == pytest.approx(0.0)
    assert ig == pytest.approx(1.0)


def test_candidate_choice_three_objects_uniform_pick_collapses_to_log2_3():
    # 3 candidates, menu with 3 obj choices + "None of them" (idx 3).
    # Uniform reply prior over 4 choices: pick i → {obj_i} (H=0); none → ∅ (H=0).
    # E[H] = 0; IG = log2(3) ≈ 1.585.
    snap = PruneSnapshot(candidates=("a", "b", "c"), excluded_obj_ids=())
    intent = PruneIntent(
        kind="candidate_choice",
        target_obj_ids=("a", "b", "c"),
        choice_to_obj={0: "a", 1: "b", 2: "c"},
        none_idx=3,
    )
    ig, h_b, h_a, per_r = information_gain_bits(intent, n_choices=4, state_before=snap)
    assert h_b == pytest.approx(math.log2(3))
    assert h_a == pytest.approx(0.0)
    assert ig == pytest.approx(math.log2(3))


def test_noop_intent_yields_zero_ig():
    snap = PruneSnapshot(candidates=("a", "b", "c"), excluded_obj_ids=())
    intent = PruneIntent(kind="noop")
    ig, h_b, h_a, per_r = information_gain_bits(intent, n_choices=2, state_before=snap)
    assert h_b == pytest.approx(math.log2(3))
    assert h_a == pytest.approx(math.log2(3))   # nothing changed
    assert ig == 0.0


def test_ig_is_non_negative_under_any_intent():
    # Even if a reply does nothing (e.g., picking an idx that maps to no obj),
    # IG floors at 0.
    snap = PruneSnapshot(candidates=("a", "b"), excluded_obj_ids=())
    intent = PruneIntent(kind="candidate_choice", target_obj_ids=("a",), choice_to_obj={0: "a"})
    ig, *_ = information_gain_bits(intent, n_choices=2, state_before=snap)
    assert ig >= 0.0
