"""Validate that mode='none' is a true byte-identical passthrough.

If a future tweak ever changes the per-call output under mode='none', this
test fails — that's the regression gate for validation criterion #1
('reranker does not change ask-vs-act decisions').
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from llm.reranker.policy_wrapper import RerankerConfig, make_reranked_backend


class _StubBackend:
    """Returns pre-baked tool calls in order; records every input."""

    def __init__(self, outputs: List[Optional[Dict[str, Any]]]):
        self.outputs = list(outputs)
        self.seen: List[Dict[str, Any]] = []
        self.calls = 0

    def __call__(self, input_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self.seen.append(input_dict)
        out = self.outputs[self.calls % len(self.outputs)]
        self.calls += 1
        return out


def _input(candidates=("obj_1", "obj_2")):
    return {
        "objects": [
            {"id": "obj_1", "label": "mug", "cell": "A1", "yaw": "N", "is_held": False},
            {"id": "obj_2", "label": "bowl", "cell": "A2", "yaw": "N", "is_held": False},
        ],
        "gripper_hist": [{"cell": "B2", "yaw": "E", "z": "HIGH"}],
        "memory": {"candidates": list(candidates), "excluded_obj_ids": []},
        "user_state": {"mode": "translation"},
    }


def test_passthrough_returns_interact_unchanged():
    interact = {
        "tool": "INTERACT",
        "args": {"kind": "CONFIRM", "text": "Do you want me to approach the mug?",
                 "choices": ["1) YES", "2) NO"]},
    }
    inner = _StubBackend([interact])
    wrapped = make_reranked_backend(
        inner, model=None, tok=None, base_cfg=None,
        config=RerankerConfig(mode="none"),
    )
    out = wrapped(_input())
    assert out == interact
    assert inner.calls == 1   # inner backend called exactly once


def test_passthrough_motion_call_unchanged():
    motion = {"tool": "APPROACH", "args": {"obj": "obj_1"}}
    inner = _StubBackend([motion])
    wrapped = make_reranked_backend(
        inner, model=None, tok=None, base_cfg=None,
        config=RerankerConfig(mode="none"),
    )
    out = wrapped(_input())
    assert out == motion
    assert inner.calls == 1


def test_passthrough_none_returns_none():
    inner = _StubBackend([None])
    wrapped = make_reranked_backend(
        inner, model=None, tok=None, base_cfg=None,
        config=RerankerConfig(mode="none"),
    )
    assert wrapped(_input()) is None


def test_info_gain_does_not_touch_motion_calls():
    """Even with mode='info_gain', non-INTERACT tools must passthrough untouched
    AND must not invoke the candidate-generation path (we pass model=None)."""
    motion = {"tool": "ALIGN_YAW", "args": {"obj": "obj_2"}}
    inner = _StubBackend([motion])
    wrapped = make_reranked_backend(
        inner, model=None, tok=None, base_cfg=None,
        config=RerankerConfig(mode="info_gain", k=5),
    )
    # If the wrapper ever tries to call generate_candidates(model=None, ...),
    # this would raise — so a clean return proves the passthrough fired first.
    out = wrapped(_input())
    assert out == motion


def test_passthrough_uses_no_model_when_inner_returns_motion():
    """Defensive — even returning None / unknown tool must skip model entirely."""
    for raw in (None, {"tool": "APPROACH", "args": {"obj": "obj_1"}}, {"tool": "UNKNOWN", "args": {}}):
        inner = _StubBackend([raw])
        wrapped = make_reranked_backend(
            inner, model=None, tok=None, base_cfg=None,
            config=RerankerConfig(mode="info_gain", k=3),
        )
        out = wrapped(_input())
        assert out == raw, f"passthrough failed for raw={raw!r}: got {out!r}"
