"""Episode sampler for the pouring task.

Scene contents:

* exactly one ``pitcher`` (kind="pitcher", fill ∈ {PARTIAL, FULL}, may start
  ``is_held=True`` with prob ``pitcher_held_p``).
* 1–3 ``cup`` objects (kind="cup", fill ∈ {EMPTY, PARTIAL, FULL}). At least one
  starts EMPTY or PARTIAL so a valid pour target exists.

Hidden ground truth on the OracleState:

* ``intended_obj_id`` — the user's chosen target cup.
* ``pending_amount`` is used at oracle runtime; the *intended* amount is stored
  on this Episode as ``intended_amount`` and the user-reply simulator uses it.

Tool effects:

* ``APPROACH(obj)`` — gripper.cell = obj.cell, z = MID.
* ``ALIGN_YAW(obj)`` — gripper.yaw = obj.yaw (so a finicky cup yaw alignment is
  representable — kept for symmetry with YCB even if rarely needed for pouring).
* ``GRAB(obj)``  — sets obj.is_held=True, gripper.cell = obj.cell, z=LOW.
* ``POUR(obj, amount)`` — increments obj.fill toward FULL based on amount; the
  pitcher's fill decrements; gripper unchanged.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import grid
from . import yaw as yawlib

CUP_LABELS: Tuple[str, ...] = ("cup_A", "cup_B", "cup_C", "mug_A", "mug_B")
PITCHER_LABEL = "pitcher"

FILL_LEVELS: Tuple[str, ...] = ("EMPTY", "PARTIAL", "FULL")
POUR_AMOUNTS: Tuple[str, ...] = ("SMALL", "HALF", "FULL")


# Discrete fill arithmetic.
_FILL_INDEX = {f: i for i, f in enumerate(FILL_LEVELS)}
_AMOUNT_STEP = {"SMALL": 1, "HALF": 1, "FULL": 2}


@dataclass
class Vessel:
    id: str
    label: str
    cell: str
    yaw: str
    kind: str  # "pitcher" | "cup"
    fill: str = "EMPTY"
    is_held: bool = False

    def to_record(self) -> Dict:
        return {
            "id": self.id,
            "label": self.label,
            "cell": self.cell,
            "yaw": self.yaw,
            "is_held": self.is_held,
            "kind": self.kind,
            "fill": self.fill,
        }


@dataclass
class Pose:
    cell: str
    yaw: str
    z: str

    def to_record(self) -> Dict:
        return {"cell": self.cell, "yaw": self.yaw, "z": self.z}


def _bump_fill(level: str, delta: int) -> str:
    idx = _FILL_INDEX[level]
    new_idx = max(0, min(len(FILL_LEVELS) - 1, idx + delta))
    return FILL_LEVELS[new_idx]


class EpisodePouring:
    """Pouring analogue of ``episode.Episode`` (same surface)."""

    def __init__(
        self,
        rng: random.Random,
        episode_id: int,
        *,
        n_cups: Optional[int] = None,
        pitcher_held_p: float = 0.5,
    ):
        self.rng = rng
        self.episode_id = episode_id
        self.T = int(rng.randint(12, 20))  # slightly longer than YCB to fit the amount sub-flow.

        if n_cups is None:
            n_cups = rng.randint(1, 3)
        n_cups = max(1, min(int(n_cups), len(CUP_LABELS)))

        # Distinct cells for everything so spatial logic stays clean.
        cells_pool = list(grid.CELLS)
        rng.shuffle(cells_pool)
        if len(cells_pool) < n_cups + 1:
            # Should never happen with a 3×3 grid and ≤3 cups.
            cells_pool = (cells_pool * 2)[: n_cups + 1]

        labels = list(CUP_LABELS)
        rng.shuffle(labels)
        cup_labels = labels[:n_cups]

        objects: List[Vessel] = []
        # Pitcher first.
        pitcher = Vessel(
            id="p0",
            label=PITCHER_LABEL,
            cell=cells_pool[0],
            yaw=rng.choice(list(yawlib.YAW_BINS)),
            kind="pitcher",
            fill=rng.choice(("PARTIAL", "FULL")),
            is_held=(rng.random() < float(pitcher_held_p)),
        )
        objects.append(pitcher)

        # Cups. Ensure at least one is non-FULL (a valid pour target).
        for i in range(n_cups):
            fill = rng.choice(FILL_LEVELS)
            objects.append(
                Vessel(
                    id=f"u{i}",
                    label=cup_labels[i],
                    cell=cells_pool[1 + i],
                    yaw=rng.choice(list(yawlib.YAW_BINS)),
                    kind="cup",
                    fill=fill,
                )
            )
        cups = [o for o in objects if o.kind == "cup"]
        if all(c.fill == "FULL" for c in cups):
            rng.choice(cups).fill = rng.choice(("EMPTY", "PARTIAL"))

        self.objects = objects
        self.pitcher_id = pitcher.id

        valid_targets = [c for c in cups if c.fill != "FULL"]
        target = rng.choice(valid_targets or cups)
        self.intended_obj_id = target.id
        self.intended_amount = rng.choice(POUR_AMOUNTS)

        # If the pitcher is held, the gripper starts at the pitcher's cell.
        if pitcher.is_held:
            init_cell = pitcher.cell
            init_z = "LOW"
        else:
            init_cell = rng.choice(list(grid.CELLS))
            init_z = rng.choice(("HIGH", "MID"))
        init_pose = Pose(cell=init_cell, yaw=rng.choice(list(yawlib.YAW_BINS)), z=init_z)
        self.gripper_hist: List[Pose] = [init_pose]
        while len(self.gripper_hist) < 6:
            self.apply_user_motion()

    # ------------------------------------------------------------------ helpers

    def get_obj(self, obj_id: str) -> Vessel:
        for o in self.objects:
            if o.id == obj_id:
                return o
        raise KeyError(obj_id)

    def intended_obj(self) -> Vessel:
        return self.get_obj(self.intended_obj_id)

    def pitcher(self) -> Vessel:
        return self.get_obj(self.pitcher_id)

    def gripper_candidates(self, max_dist: int = 1) -> List[str]:
        """Plausible POUR targets: cups within ``max_dist`` that aren't already FULL."""
        cell = self.gripper_hist[-1].cell
        out: List[str] = []
        for o in self.objects:
            if o.kind != "cup" or o.fill == "FULL":
                continue
            if grid.manhattan(cell, o.cell) <= max_dist:
                out.append(o.id)
        return out

    def _push_gripper(self, pose: Pose) -> None:
        self.gripper_hist.append(pose)
        if len(self.gripper_hist) > 6:
            self.gripper_hist = self.gripper_hist[-6:]

    # ----------------------------------------------------------- user motion

    def apply_user_motion(self) -> None:
        cur = self.gripper_hist[-1]
        pitcher = self.pitcher()
        intended = self.intended_obj()

        # If the pitcher isn't held yet, the user's intent is the pitcher cell.
        # Otherwise, intent is the target cup.
        goal_cell = pitcher.cell if not pitcher.is_held else intended.cell

        if self.rng.random() < 0.8:
            next_cell = grid.step_toward(cur.cell, goal_cell)
        else:
            neigh = grid.neighbors(cur.cell)
            next_cell = self.rng.choice(neigh) if neigh else cur.cell

        # Yaw drift toward intended yaw (target object).
        if cur.cell == intended.cell and self.rng.random() < 0.45:
            yn = yawlib.neighbors(intended.yaw)
            next_yaw = self.rng.choice([cur.yaw, yn[0], yn[1]])
        else:
            next_yaw = yawlib.move_toward(cur.yaw, intended.yaw, steps=1)

        # Z behaviour: dip when reaching pitcher to grab; hover above cups; dip on arrival.
        if next_cell == pitcher.cell and not pitcher.is_held:
            next_z = "LOW" if cur.z != "HIGH" else "MID"
        elif next_cell == intended.cell:
            next_z = "MID" if cur.z == "HIGH" else "LOW"
        else:
            next_z = "HIGH" if cur.z == "LOW" else cur.z

        self._push_gripper(Pose(cell=next_cell, yaw=next_yaw, z=next_z))

    # -------------------------------------------------------------- tool apply

    def apply_tool(self, tool_call: Dict) -> None:
        tool = tool_call["tool"]
        args = tool_call.get("args") or {}
        cur = self.gripper_hist[-1]

        def set_gripper(cell: Optional[str] = None, yaw: Optional[str] = None, z: Optional[str] = None) -> None:
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
            set_gripper(cell=obj.cell, z="MID")
            return
        if tool == "ALIGN_YAW":
            obj = self.get_obj(args["obj"])
            set_gripper(yaw=obj.yaw)
            return
        if tool == "GRAB":
            obj = self.get_obj(args["obj"])
            obj.is_held = True
            set_gripper(cell=obj.cell, z="LOW")
            return
        if tool == "POUR":
            cup = self.get_obj(args["obj"])
            amount = args.get("amount", "HALF")
            step = _AMOUNT_STEP.get(amount, 1)
            cup.fill = _bump_fill(cup.fill, step)
            self.pitcher().fill = _bump_fill(self.pitcher().fill, -step)
            return
        raise ValueError(f"Unknown pouring tool: {tool}")
