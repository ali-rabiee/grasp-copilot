"""
Low-level parsers for a single PRIME_LOGS trial directory.

Each trial directory contains:
  trial_meta.json       — start_time, mode, difficulty, subject_id, ...
  trial_summary.json    — counts, durations (success fields are unreliable —
                          empty for every trial we have)
  events.jsonl          — union of all events (gui + tool + ...)
  gui_events.jsonl      — raw user velocity / mode commands (per-line {parsed,raw})
  tool_calls.jsonl      — PRIME tool calls with state_snapshot.* (assistive only)
  tool_results.jsonl    — outcomes of tool calls
  queries.jsonl         — PRIME INTERACT prompts shown to the user (usually empty)
  responses.jsonl       — user answers to those prompts (usually empty)

This module provides only data access; interpretation (target inference, scene
templates, priors) lives in dedicated modules.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RawObject:
    """An object's state inside a single tool_calls.jsonl state_snapshot."""

    id: str
    label: str
    grid_cell: int
    grid_label: str
    is_held: bool
    yaw_rad: float = 0.0


@dataclass
class RawGripper:
    grid_cell: int
    grid_label: str
    yaw_rad: float
    height_m: float


@dataclass
class StateSnapshot:
    objects: List[RawObject]
    gripper: RawGripper
    stamp: float


@dataclass
class ToolCallRecord:
    t_from_start_sec: float
    tool_name: str
    target_object_id: Optional[str]
    state_snapshot: StateSnapshot


@dataclass
class TrialData:
    trial_dir: Path
    meta: Dict[str, Any]
    summary: Dict[str, Any]
    gui_events: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[ToolCallRecord] = field(default_factory=list)


# ── helpers ─────────────────────────────────────────────────────────────


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _parse_state_snapshot(snap: Dict[str, Any]) -> StateSnapshot:
    raw_objs = []
    for o in snap.get("objects", []) or []:
        raw_objs.append(
            RawObject(
                id=str(o.get("id", "")),
                label=str(o.get("label", "")),
                grid_cell=int(o.get("grid_cell", 0)),
                grid_label=str(o.get("grid_label", "")),
                is_held=bool(o.get("is_held", False)),
                yaw_rad=float(o.get("yaw", 0.0)),
            )
        )
    g = snap.get("gripper", {}) or {}
    raw_g = RawGripper(
        grid_cell=int(g.get("grid_cell", 0)),
        grid_label=str(g.get("grid_label", "")),
        yaw_rad=float(g.get("yaw", 0.0)),
        height_m=float(g.get("height", 0.0)),
    )
    return StateSnapshot(objects=raw_objs, gripper=raw_g, stamp=float(snap.get("stamp", 0.0)))


def _parse_tool_call(tc: Dict[str, Any]) -> Optional[ToolCallRecord]:
    snap_dict = tc.get("state_snapshot")
    if not isinstance(snap_dict, dict):
        return None
    snap = _parse_state_snapshot(snap_dict)
    tgt = tc.get("target_object_id") or None
    return ToolCallRecord(
        t_from_start_sec=float(tc.get("t_from_start_sec", 0.0)),
        tool_name=str(tc.get("tool_name", "")),
        target_object_id=tgt if tgt else None,
        state_snapshot=snap,
    )


# ── public API ──────────────────────────────────────────────────────────


def load_trial(trial_dir: str | Path) -> Optional[TrialData]:
    """Read a single trial directory. Returns None only if trial_summary.json is missing.

    Some trials in PRIME_LOGS never wrote trial_meta.json (26 of 160). Since
    trial_summary.json carries the same identifying fields (mode, subject_id,
    difficulty, trial_id), we fall back to it when meta is missing rather than
    drop the whole trial.
    """
    p = Path(trial_dir)
    meta_path = p / "trial_meta.json"
    summary_path = p / "trial_summary.json"
    if not summary_path.exists():
        return None

    summary = _read_json(summary_path)
    if meta_path.exists():
        meta = _read_json(meta_path)
    else:
        meta = {
            "mode": summary.get("mode"),
            "subject_id": summary.get("subject_id"),
            "difficulty": summary.get("difficulty"),
            "trial_id": summary.get("trial_id"),
            "_meta_synthesized_from_summary": True,
        }

    gui_events: List[Dict[str, Any]] = []
    for ev in _iter_jsonl(p / "gui_events.jsonl"):
        parsed = ev.get("parsed", ev)
        if isinstance(parsed, dict):
            gui_events.append(parsed)

    tool_calls: List[ToolCallRecord] = []
    for tc in _iter_jsonl(p / "tool_calls.jsonl"):
        rec = _parse_tool_call(tc)
        if rec is not None:
            tool_calls.append(rec)

    return TrialData(trial_dir=p, meta=meta, summary=summary, gui_events=gui_events, tool_calls=tool_calls)


def discover_trials(logs_root: str | Path) -> List[Tuple[str, str, str, Path]]:
    """Yield (mode, subject, difficulty, trial_dir) for every trial directory.

    Skips the `old/` subdirectory (pilot trials) and any trial dir without a
    trial_summary.json.
    """
    root = Path(logs_root)
    out: List[Tuple[str, str, str, Path]] = []
    for mode in ("manual", "assistive"):
        mode_dir = root / mode
        if not mode_dir.is_dir():
            continue
        for subject_dir in sorted(mode_dir.iterdir()):
            if not subject_dir.is_dir() or not subject_dir.name.startswith("s"):
                continue
            for difficulty in ("easy", "hard"):
                diff_dir = subject_dir / difficulty
                if not diff_dir.is_dir():
                    continue
                for trial_dir in sorted(diff_dir.iterdir()):
                    if not trial_dir.is_dir():
                        continue
                    if not trial_dir.name.startswith("trial_"):
                        continue
                    if not (trial_dir / "trial_summary.json").exists():
                        continue
                    out.append((mode, subject_dir.name, difficulty, trial_dir))
    return out


# ── duplicate-detection cleanup ─────────────────────────────────────────


def dedup_objects(objs: List[RawObject]) -> List[RawObject]:
    """Drop duplicate detections of the same physical object.

    Camera perception in the deployed system occasionally double-detects an
    object (e.g. `coffee_can@A1` appears twice). We collapse duplicates by
    (canonical label, cell), keeping the first occurrence and *reissuing*
    object ids contiguously as `obj_1`, `obj_2`, ... so downstream code can
    rely on stable id semantics.

    The is_held flag wins on logical OR — if any detection of the object is
    held, the merged record is held.
    """
    seen: Dict[Tuple[str, str], RawObject] = {}
    order: List[Tuple[str, str]] = []
    for o in objs:
        key = (o.label, o.grid_label)
        if key in seen:
            existing = seen[key]
            if o.is_held and not existing.is_held:
                existing.is_held = True
            continue
        seen[key] = RawObject(
            id="",  # reassigned below
            label=o.label,
            grid_cell=o.grid_cell,
            grid_label=o.grid_label,
            is_held=o.is_held,
            yaw_rad=o.yaw_rad,
        )
        order.append(key)

    out: List[RawObject] = []
    for i, key in enumerate(order, start=1):
        merged = seen[key]
        merged.id = f"obj_{i}"
        out.append(merged)
    return out


__all__ = [
    "RawObject",
    "RawGripper",
    "StateSnapshot",
    "ToolCallRecord",
    "TrialData",
    "discover_trials",
    "load_trial",
    "dedup_objects",
]
