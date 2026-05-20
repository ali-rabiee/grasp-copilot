"""
Scenario record schema — the contract every downstream component reads.

A scenario is fully defined by:
  * the initial 3x3 grid layout of objects (id, normalized label, cell, yaw bin),
  * the initial gripper pose (cell, yaw bin, z bin),
  * the user's intended target object id (may be None until labeled),
  * behavioral priors derived from the trial's gui_events.jsonl.

The schema is JSONL-serializable and uses primitive types only.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

# ── canonical vocab (mirrors data_generator/grid.py + yaw.py) ────────────
CELLS = tuple(f"{r}{c}" for r in "ABC" for c in (1, 2, 3))  # A1..C3
YAW_BINS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
Z_BINS = ("HIGH", "MID", "LOW")

# Source of the target label — paper text must say which provenance dominates.
TARGET_SOURCE_TOOL_CALL = "tool_call"          # last APPROACH/ALIGN_YAW target
TARGET_SOURCE_HAND_LABEL = "hand_label"        # from manual_targets.csv
TARGET_SOURCE_UNLABELED = "unlabeled"          # awaiting hand-label
ALL_TARGET_SOURCES = (
    TARGET_SOURCE_TOOL_CALL,
    TARGET_SOURCE_HAND_LABEL,
    TARGET_SOURCE_UNLABELED,
)

# Source of the object layout.
LAYOUT_SOURCE_STATE_SNAPSHOT = "state_snapshot"   # from assistive tool_calls.jsonl
LAYOUT_SOURCE_BORROWED_TEMPLATE = "borrowed_template"  # canonical (subj,diff) scene
LAYOUT_SOURCE_HAND_LABEL = "hand_label"           # from manual_targets.csv
ALL_LAYOUT_SOURCES = (
    LAYOUT_SOURCE_STATE_SNAPSHOT,
    LAYOUT_SOURCE_BORROWED_TEMPLATE,
    LAYOUT_SOURCE_HAND_LABEL,
)


@dataclass
class ObjectInit:
    """An object's initial pose in the 3x3 grid."""

    id: str
    label: str            # canonical label (after normalize)
    raw_label: str        # original PRIME_LOGS label, for traceability
    cell: str             # A1..C3
    yaw: str              # one of YAW_BINS
    is_held: bool = False


@dataclass
class GripperInit:
    cell: str
    yaw: str
    z: str = "HIGH"       # gripper starts at home height in every trial


@dataclass
class UserPriors:
    """Behavioral priors computed from gui_events.jsonl.

    All shares sum to 1 if any command was emitted; rates are per second of
    active task time (i.e. from first command to last command).
    """

    translation_share: float = 0.0
    rotation_share: float = 0.0
    gripper_share: float = 0.0
    mean_active_burst_sec: float = 0.0
    mode_switches_per_sec: float = 0.0
    direction_reversals_per_sec: float = 0.0
    total_active_time_sec: float = 0.0
    total_commands: int = 0


@dataclass
class TrialSource:
    """Provenance pointer back to the original PRIME_LOGS trial."""

    mode: str            # "manual" | "assistive"
    subject: str         # "s1".."s8"
    difficulty: str      # "easy" | "hard"
    trial_id: str        # "trial_YYYYMMDD_HHMMSS"
    trial_dir: str       # repo-relative path


@dataclass
class Scenario:
    scenario_id: str
    source: TrialSource
    objects: List[ObjectInit]
    gripper_init: GripperInit
    target_obj_id: Optional[str]            # None until labeled
    target_label_source: str                # ALL_TARGET_SOURCES
    layout_source: str                      # ALL_LAYOUT_SOURCES
    user_priors: UserPriors
    difficulty: str                         # easy | hard (mirrors source.difficulty for filtering)
    plausible_target_ids: List[str] = field(default_factory=list)
    notes: str = ""

    # ── validation ───────────────────────────────────────────────────────

    def validate(self) -> List[str]:
        """Return a list of problems. Empty list = valid."""
        problems: List[str] = []

        if not self.objects:
            problems.append("no objects")
        if len(self.objects) < 2:
            problems.append(f"need >=2 objects, have {len(self.objects)}")

        obj_ids = [o.id for o in self.objects]
        if len(set(obj_ids)) != len(obj_ids):
            problems.append(f"duplicate object ids: {obj_ids}")

        for o in self.objects:
            if o.cell not in CELLS:
                problems.append(f"object {o.id} bad cell {o.cell}")
            if o.yaw not in YAW_BINS:
                problems.append(f"object {o.id} bad yaw {o.yaw}")

        if self.gripper_init.cell not in CELLS:
            problems.append(f"gripper bad cell {self.gripper_init.cell}")
        if self.gripper_init.yaw not in YAW_BINS:
            problems.append(f"gripper bad yaw {self.gripper_init.yaw}")
        if self.gripper_init.z not in Z_BINS:
            problems.append(f"gripper bad z {self.gripper_init.z}")

        if self.target_obj_id is not None and self.target_obj_id not in obj_ids:
            problems.append(f"target {self.target_obj_id} not in objects {obj_ids}")
        if self.target_label_source not in ALL_TARGET_SOURCES:
            problems.append(f"bad target_label_source {self.target_label_source}")
        if (self.target_obj_id is None) != (self.target_label_source == TARGET_SOURCE_UNLABELED):
            problems.append(
                f"target/source inconsistent: id={self.target_obj_id} src={self.target_label_source}"
            )

        if self.layout_source not in ALL_LAYOUT_SOURCES:
            problems.append(f"bad layout_source {self.layout_source}")

        for pid in self.plausible_target_ids:
            if pid not in obj_ids:
                problems.append(f"plausible target {pid} not in objects")

        return problems


# ── JSONL serialization ─────────────────────────────────────────────────


def _scenario_to_dict(s: Scenario) -> Dict[str, Any]:
    return asdict(s)


def _scenario_from_dict(d: Dict[str, Any]) -> Scenario:
    return Scenario(
        scenario_id=d["scenario_id"],
        source=TrialSource(**d["source"]),
        objects=[ObjectInit(**o) for o in d["objects"]],
        gripper_init=GripperInit(**d["gripper_init"]),
        target_obj_id=d.get("target_obj_id"),
        target_label_source=d["target_label_source"],
        layout_source=d["layout_source"],
        user_priors=UserPriors(**d["user_priors"]),
        difficulty=d["difficulty"],
        plausible_target_ids=list(d.get("plausible_target_ids", [])),
        notes=d.get("notes", ""),
    )


def write_scenarios(path: str | Path, scenarios: Sequence[Scenario]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for s in scenarios:
            f.write(json.dumps(_scenario_to_dict(s), ensure_ascii=False) + "\n")


def load_scenarios(path: str | Path) -> List[Scenario]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [_scenario_from_dict(json.loads(line)) for line in f if line.strip()]


def iter_scenarios(path: str | Path) -> Iterator[Scenario]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield _scenario_from_dict(json.loads(line))
