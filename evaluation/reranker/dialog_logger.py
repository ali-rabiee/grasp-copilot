"""JSONL append-logger for reranker IG decisions.

One line per emitted INTERACT call. Schema matches plan §4. The file is
opened in append mode so multiple processes / re-runs accumulate (the
runner truncates it before the sweep starts).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


_LOCK = threading.Lock()


def _state_hash(input_dict: Dict[str, Any]) -> str:
    """Cheap stable hash of (objects, gripper_hist, memory.candidates) for dedupe / oracle lookup."""
    objs = sorted(
        (str(o.get("id", "")), str(o.get("cell", "")), str(o.get("yaw", "")))
        for o in (input_dict.get("objects") or [])
    )
    gh = input_dict.get("gripper_hist") or []
    cur = gh[-1] if gh else {}
    mem = input_dict.get("memory") or {}
    cands = sorted(str(c) for c in (mem.get("candidates") or []))
    excl = sorted(str(c) for c in (mem.get("excluded_obj_ids") or []))
    payload = json.dumps(
        {"objs": objs, "gripper": [cur.get("cell"), cur.get("yaw")], "cands": cands, "excl": excl},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class DialogLogger:
    path: Path
    scenario_id: str = ""
    seed: int = 0
    condition: str = ""
    tick: int = 0
    _fh: Optional[Any] = field(default=None, init=False, repr=False)

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
            except Exception:
                pass
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "DialogLogger":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def set_episode(self, scenario_id: str, seed: int, condition: str) -> None:
        self.scenario_id = scenario_id
        self.seed = int(seed)
        self.condition = condition
        self.tick = 0

    def tick_inc(self) -> None:
        self.tick += 1

    def log(
        self,
        *,
        input_dict: Dict[str, Any],
        selector_name: str,
        chosen: Dict[str, Any],
        chosen_ig: float,
        h_before: float,
        h_after_expected: float,
        candidates: Sequence[Any],
        per_candidate_scores: Sequence[float],
        per_candidate_details: Sequence[List],
        n_candidates_before: int,
        context_type_hint: str = "",
    ) -> None:
        if self._fh is None:
            return
        args = chosen.get("args") or {}
        rec = {
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "condition": self.condition,
            "tick": self.tick,
            "state_hash": _state_hash(input_dict),
            "selector": selector_name,
            "k_candidates": len(candidates),
            "n_candidates_before": int(n_candidates_before),
            "h_before_bits": round(float(h_before), 6),
            "h_after_expected_bits": round(float(h_after_expected), 6),
            "ig_bits": round(float(chosen_ig), 6),
            "interact_kind": str(args.get("kind", "")).upper(),
            "context_type_hint": context_type_hint,
            "chosen": chosen,
            "candidates": [
                {
                    "sample_idx": getattr(c, "sample_idx", i),
                    "tool_call": getattr(c, "tool_call", c),
                    "ig_bits": round(float(per_candidate_scores[i]) if i < len(per_candidate_scores) else 0.0, 6),
                    "per_reply": [
                        [int(r[0]), round(float(r[1]), 6), round(float(r[2]), 6)]
                        for r in (per_candidate_details[i] if i < len(per_candidate_details) else [])
                    ],
                }
                for i, c in enumerate(candidates)
            ],
        }
        with _LOCK:
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fh.flush()
