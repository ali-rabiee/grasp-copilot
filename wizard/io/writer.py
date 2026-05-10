"""JSONL writer for wizard episodes.

Output line schema matches ``data/runs/*/grasp_gen.jsonl`` exactly so the
existing ``llm.prepare_llm_data`` pipeline can consume WoZ data with no
changes:

    {"episode_id": ..., "objects": [...], "gripper_hist": [...],
     "memory": {...}, "user_state": {...},
     "target_tool_call": {"tool": ..., "args": ...}}

We also write per-episode metadata (wizard id, intended object id, alert
reasons, env name) to a sidecar ``episodes_meta.jsonl`` so we can compute
inter-wizard agreement and ambiguity-stratified scores later. Ground-truth
fields live ONLY in this sidecar — the trainable JSONL never sees them.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from data_generator.oracle import validate_tool_call


@dataclass
class EpisodeWriter:
    """Per-run writer. One run = one wizard × one collection session."""

    out_dir: Path
    wizard_id: str
    run_name: str = "woz_001"

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._train_path = self.out_dir / "grasp_gen.jsonl"
        self._meta_path = self.out_dir / "episodes_meta.jsonl"
        self._train_f = open(self._train_path, "a", buffering=1)
        self._meta_f = open(self._meta_path, "a", buffering=1)
        self._current_meta: Optional[Dict] = None
        self._decisions_in_episode: int = 0

    # --------------------------------------------------------------- lifecycle

    def start_episode(self, episode_id: int, env_name: str, wizard_id: str,
                      intended_obj_id: str) -> None:
        self._current_meta = {
            "episode_id": episode_id,
            "env_name": env_name,
            "wizard_id": wizard_id,
            "intended_obj_id": intended_obj_id,
            "started_at": time.time(),
            "decisions": [],
        }
        self._decisions_in_episode = 0

    def write_decision(self, blob: Dict, tool_call: Dict, alert_reason: str) -> None:
        validate_tool_call(tool_call)
        if self._current_meta is None:
            raise RuntimeError("write_decision called before start_episode")
        episode_id = self._current_meta["episode_id"]

        line = {
            "episode_id": episode_id,
            "objects": blob["objects"],
            "gripper_hist": blob["gripper_hist"],
            "memory": blob["memory"],
            "user_state": blob["user_state"],
            "target_tool_call": tool_call,
        }
        self._train_f.write(json.dumps(line, ensure_ascii=False) + "\n")

        self._current_meta["decisions"].append({
            "tick_idx": self._decisions_in_episode,
            "alert_reason": alert_reason,
            "tool_call": tool_call,
        })
        self._decisions_in_episode += 1

    def end_episode(self) -> None:
        if self._current_meta is None:
            return
        self._current_meta["ended_at"] = time.time()
        self._current_meta["n_decisions"] = self._decisions_in_episode
        self._meta_f.write(json.dumps(self._current_meta, ensure_ascii=False) + "\n")
        self._current_meta = None
        self._decisions_in_episode = 0

    def close(self) -> None:
        try:
            self._train_f.flush()
            self._train_f.close()
        except Exception:
            pass
        try:
            self._meta_f.flush()
            self._meta_f.close()
        except Exception:
            pass
