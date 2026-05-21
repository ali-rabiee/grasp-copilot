from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from . import grid
from . import yaw as yawlib


OBJECT_LABELS: Tuple[str, ...] = (
    "bleach_cleanser",
    "gelatin_box",
    "master_chef_can",
    "mug",
    "mustard_bottle",
    "potted_meat_can",
    "pudding_box",
    "sugar_box",
    "tomato_soup_can",
    "tuna_fish_can",
)

Z_BINS: Tuple[str, ...] = ("HIGH", "MID", "LOW")


@dataclass
class Obj:
    id: str
    label: str
    cell: str
    yaw: str
    is_held: bool = False

    def to_record(self) -> Dict:
        return {"id": self.id, "label": self.label, "cell": self.cell, "yaw": self.yaw, "is_held": self.is_held}


@dataclass
class Pose:
    cell: str
    yaw: str
    z: str

    def to_record(self) -> Dict:
        return {"cell": self.cell, "yaw": self.yaw, "z": self.z}


class Episode:
    """
    Lightweight environment simulator that keeps gripper history and object layout.

    The episode never executes a true grasp; it only simulates intent-driven motion
    so the oracle can emit INTERACT / APPROACH / ALIGN_YAW tool calls.
    """

    def __init__(self, rng, episode_id: int, n_obj: int, collision_p: float = 0.15):
        if not (2 <= n_obj <= len(OBJECT_LABELS)):
            raise ValueError("n_obj must be in [2, len(OBJECT_LABELS)]")
        self.rng = rng
        self.episode_id = episode_id
        self.T = int(rng.randint(10, 18))

        labels = list(OBJECT_LABELS)
        rng.shuffle(labels)
        labels = labels[:n_obj]

        objects: List[Obj] = []
        for i, label in enumerate(labels):
            obj_id = f"o{i}"
            if objects and rng.random() < collision_p:
                cell = rng.choice([o.cell for o in objects])
            else:
                cell = rng.choice(list(grid.CELLS))
            yaw = rng.choice(list(yawlib.YAW_BINS))
            objects.append(Obj(id=obj_id, label=label, cell=cell, yaw=yaw))

        self.objects = objects
        self.intended_obj_id: str = rng.choice([o.id for o in objects])

        init_pose = Pose(
            cell=rng.choice(list(grid.CELLS)),
            yaw=rng.choice(list(yawlib.YAW_BINS)),
            z=rng.choice(Z_BINS),
        )
        # Option B: seed with a short "teleop trajectory" toward the intended object,
        # so the initial history contains motion evidence the model can reason over.
        # We still keep exactly 6 poses.
        self.gripper_hist: List[Pose] = [init_pose]
        while len(self.gripper_hist) < 6:
            self.apply_user_motion()

    def get_obj(self, obj_id: str) -> Obj:
        for o in self.objects:
            if o.id == obj_id:
                return o
        raise KeyError(obj_id)

    def intended_obj(self) -> Obj:
        return self.get_obj(self.intended_obj_id)

    def gripper_candidates(self, max_dist: int = 1) -> List[str]:
        """
        Returns object ids that are within <= max_dist Manhattan distance of the gripper.
        """
        cell = self.gripper_hist[-1].cell
        out: List[str] = []
        for o in self.objects:
            if o.is_held:
                continue
            if grid.manhattan(cell, o.cell) <= max_dist:
                out.append(o.id)
        return out

    def _push_gripper(self, pose: Pose) -> None:
        self.gripper_hist.append(pose)
        if len(self.gripper_hist) > 6:
            self.gripper_hist = self.gripper_hist[-6:]

    def apply_user_motion(self) -> None:
        """
        Simulate noisy human teleop intent toward the intended object.
        """
        cur = self.gripper_hist[-1]
        intended = self.intended_obj()

        # Cell motion: mostly step toward intent, sometimes jitter to a neighbor.
        if self.rng.random() < 0.8:
            next_cell = grid.step_toward(cur.cell, intended.cell)
        else:
            neigh = grid.neighbors(cur.cell)
            next_cell = self.rng.choice(neigh) if neigh else cur.cell

        # Yaw motion: mix of direct alignment and oscillation when on target cell.
        if cur.cell == intended.cell and self.rng.random() < 0.55:
            # Deliberately oscillate to create yaw-struggle cases.
            yaw_neighbors = yawlib.neighbors(intended.yaw)
            next_yaw = self.rng.choice([cur.yaw, yaw_neighbors[0], yaw_neighbors[1]])
        else:
            next_yaw = yawlib.move_toward(cur.yaw, intended.yaw, steps=1)

        # Z motion trends down when close to target, otherwise hovers higher.
        if next_cell == intended.cell:
            if cur.z == "HIGH":
                next_z = "MID" if self.rng.random() < 0.7 else "HIGH"
            elif cur.z == "MID":
                next_z = "LOW" if self.rng.random() < 0.65 else "MID"
            else:
                next_z = "LOW"
        else:
            if cur.z == "LOW":
                next_z = "MID" if self.rng.random() < 0.6 else "LOW"
            elif cur.z == "MID":
                next_z = "HIGH" if self.rng.random() < 0.55 else "MID"
            else:
                next_z = "HIGH"

        self._push_gripper(Pose(cell=next_cell, yaw=next_yaw, z=next_z))

    def apply_tool(self, tool_call: Dict) -> None:
        """
        Apply a tool call to the simulated environment (no grasp execution).
        """
        tool = tool_call["tool"]
        args = tool_call["args"]
        cur = self.gripper_hist[-1]

        def set_gripper(cell: Optional[str] = None, yaw: Optional[str] = None, z: Optional[str] = None):
            self._push_gripper(
                Pose(
                    cell=cell if cell is not None else cur.cell,
                    yaw=yaw if yaw is not None else cur.yaw,
                    z=z if z is not None else cur.z,
                )
            )

        if tool == "INTERACT":
            return
        if tool == "APPROACH":
            obj = self.get_obj(args["obj"])
            set_gripper(cell=obj.cell, z="HIGH")
        elif tool == "ALIGN_YAW":
            obj = self.get_obj(args["obj"])
            set_gripper(yaw=obj.yaw)
        else:
            raise ValueError(f"Unknown tool: {tool}")

    # ── scenario-seeded + target-agnostic stepping (noise-robustness rollouts) ──

    @classmethod
    def from_scenario(cls, scenario, rng) -> "Episode":
        """Construct an Episode from a Scenario record without random sampling.

        Used by the noise-robustness rollout driver (`evaluation/rollouts/`).
        Skips `__init__` (which is random-sampling-based and target-aware) and
        instead builds an Episode whose state mirrors the scenario's initial
        conditions exactly:

          * objects, gripper pose, target object come from the scenario
          * gripper_hist is seeded with 6 copies of the initial pose (signals
            "no motion yet" — matches the convention in `evaluation/scenarios/
            adapters.scenario_to_input_dict`)
          * gripper_closed starts False
          * episode_id is the scenario_id, T is left as a soft horizon for the
            caller to override

        Accepts either a dataclass `Scenario` object or a plain dict with the
        same field names.
        """
        ep = cls.__new__(cls)
        ep.rng = rng

        def _attr(obj, name, default=None):
            if hasattr(obj, name):
                return getattr(obj, name)
            if isinstance(obj, dict):
                return obj.get(name, default)
            return default

        ep.episode_id = _attr(scenario, "scenario_id", 0)
        ep.T = 200   # soft horizon for the rollout caller to cap

        ep.objects = []
        for o in _attr(scenario, "objects", []) or []:
            ep.objects.append(
                Obj(
                    id=_attr(o, "id"),
                    label=_attr(o, "label"),
                    cell=_attr(o, "cell"),
                    yaw=_attr(o, "yaw"),
                    is_held=bool(_attr(o, "is_held", False)),
                )
            )

        target = _attr(scenario, "target_obj_id", None)
        ep.intended_obj_id = target if target is not None else (ep.objects[0].id if ep.objects else "")

        g = _attr(scenario, "gripper_init")
        init_pose = Pose(
            cell=_attr(g, "cell"),
            yaw=_attr(g, "yaw"),
            z=_attr(g, "z", "HIGH"),
        )
        ep.gripper_hist = [init_pose for _ in range(6)]

        ep.gripper_closed = False
        return ep

    def step_user_command(self, axis: str, direction: int, mode: str) -> None:
        """Apply one target-agnostic user velocity tick to the gripper state.

        Mirrors the per-tick GUI commands recorded in PRIME_LOGS gui_events:
        the user pushes a joystick / head-array key, and the gripper moves
        one discrete step. Boundary moves are clamped (real teleop just
        doesn't move past the workspace edge).

        Conventions:
          mode="translation":
              axis="x" → column (cells …1, …2, …3); +1 increases column
              axis="y" → row (A…, B…, C…);          +1 increases row
              axis="z" → height bin;                 +1 goes HIGH-ward
          mode="rotation":
              axis is ignored (we have a single 8-bin yaw ring);
              +1 walks one yaw bin clockwise per `yaw.move_toward` semantics
          mode="gripper":
              toggles `gripper_closed`; direction is ignored.
        """
        if direction not in (-1, 1) and mode != "gripper":
            raise ValueError(f"direction must be -1 or +1 (got {direction!r})")

        cur = self.gripper_hist[-1]
        new_cell, new_yaw, new_z = cur.cell, cur.yaw, cur.z

        if mode == "translation":
            if axis == "x":
                new_cell = _step_cell(cur.cell, drow=0, dcol=direction)
            elif axis == "y":
                new_cell = _step_cell(cur.cell, drow=direction, dcol=0)
            elif axis == "z":
                new_z = _step_z(cur.z, direction)
            else:
                raise ValueError(f"unknown translation axis {axis!r}")
        elif mode == "rotation":
            new_yaw = _step_yaw(cur.yaw, direction)
        elif mode == "gripper":
            self.gripper_closed = not getattr(self, "gripper_closed", False)
            # gripper toggle doesn't change pose — no history push.
            return
        else:
            raise ValueError(f"unknown mode {mode!r}")

        self._push_gripper(Pose(cell=new_cell, yaw=new_yaw, z=new_z))


# ── helpers for step_user_command (kept module-private) ────────────────────


def _step_cell(cell: str, drow: int, dcol: int) -> str:
    """Move one grid step; clamp at boundaries."""
    c = grid.Cell.from_label(cell)
    nr = max(0, min(2, c.r + drow))
    nc = max(0, min(2, c.c + dcol))
    return grid.Cell(nr, nc).to_label()


def _step_z(z: str, direction: int) -> str:
    """+1 → toward HIGH; -1 → toward LOW. Clamp at endpoints."""
    idx = Z_BINS.index(z)
    if direction > 0:
        idx = max(0, idx - 1)   # HIGH is at index 0, so + goes toward 0
    else:
        idx = min(len(Z_BINS) - 1, idx + 1)
    return Z_BINS[idx]


def _step_yaw(yaw: str, direction: int) -> str:
    """One bin step on the 8-bin yaw ring (no clamping; wraps around)."""
    bins = yawlib.YAW_BINS
    idx = bins.index(yaw)
    return bins[(idx + (1 if direction > 0 else -1)) % len(bins)]


def write_jsonl(path: str, records: Sequence[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

