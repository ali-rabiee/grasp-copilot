"""Selectors pick one of K scored candidate questions.

Four flavors:
  InfoGainSelector — argmax IG; tie-break by smaller |choices| (cheaper question).
  RandomSelector   — uniform random over candidates (control).
  OracleSelector   — picks the candidate closest in JSON to the oracle's emit.
  NoneSelector     — always returns index 0 (passthrough: the base LLM's call).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence


class Selector(Protocol):
    name: str
    def pick(self, candidates: Sequence[Any], scored: Sequence[float]) -> int: ...


@dataclass
class InfoGainSelector:
    name: str = "info_gain"

    def pick(self, candidates: Sequence[Any], scored: Sequence[float]) -> int:
        if not candidates:
            return 0
        best_i = 0
        best_ig = float("-inf")
        best_len = float("inf")
        for i, c in enumerate(candidates):
            ig = float(scored[i]) if i < len(scored) else 0.0
            n_ch = len((c.tool_call.get("args") or {}).get("choices") or [])
            if ig > best_ig or (ig == best_ig and n_ch < best_len):
                best_i, best_ig, best_len = i, ig, n_ch
        return best_i


@dataclass
class RandomSelector:
    name: str = "random"
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def pick(self, candidates: Sequence[Any], scored: Sequence[float]) -> int:
        if not candidates:
            return 0
        return self._rng.randrange(len(candidates))


@dataclass
class NoneSelector:
    name: str = "none"

    def pick(self, candidates: Sequence[Any], scored: Sequence[float]) -> int:
        return 0


@dataclass
class OracleSelector:
    """Picks the candidate closest in JSON to the oracle's emit at the current state.

    Used at offline-analysis time. At online sweep time, building a fresh
    OracleState per call is feasible but expensive — prefer running this
    offline against dialogs.jsonl.
    """
    name: str = "oracle"
    oracle_emit: Optional[Dict[str, Any]] = None

    def set_oracle_emit(self, emit: Optional[Dict[str, Any]]) -> None:
        self.oracle_emit = emit

    @staticmethod
    def _stringify(d: Optional[Dict]) -> str:
        if not d:
            return ""
        try:
            return json.dumps(d, sort_keys=True, ensure_ascii=False)
        except Exception:
            return str(d)

    def pick(self, candidates: Sequence[Any], scored: Sequence[float]) -> int:
        if not candidates:
            return 0
        target = self._stringify(self.oracle_emit)
        if not target:
            return 0
        best_i = 0
        best_overlap = -1
        for i, c in enumerate(candidates):
            s = self._stringify(c.tool_call)
            # Cheap similarity: count shared tokens between args.choices.
            ca = (c.tool_call.get("args") or {}).get("choices") or []
            oa = ((self.oracle_emit or {}).get("args") or {}).get("choices") or []
            overlap = len(set(map(str, ca)) & set(map(str, oa)))
            # Tie-breaker: identical JSON wins.
            if s == target:
                return i
            if overlap > best_overlap:
                best_i, best_overlap = i, overlap
        return best_i


def make_selector(name: str, *, seed: int = 0) -> Selector:
    n = (name or "").strip().lower()
    if n == "info_gain":
        return InfoGainSelector()
    if n == "random":
        return RandomSelector(seed=seed)
    if n == "oracle":
        return OracleSelector()
    if n in ("none", "passthrough"):
        return NoneSelector()
    raise ValueError(f"unknown selector: {name!r}")
