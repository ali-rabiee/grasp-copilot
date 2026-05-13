"""Episode sampler for the cube-stacking task.

A small set of colored cubes sits on the workspace. Exactly one cube starts
**held** in the gripper. 0-2 of the table cubes may have a second cube already
stacked on them. The hidden ground truth is the user's chosen ``intended_base_id``
— the cube they want their held cube placed on top of.

Per-object fields (in addition to the standard YCB-style {id, label, cell, yaw, is_held}):

* ``stacked_on``  — id of the cube directly beneath this one, or None.
* ``top_of_stack`` — True if no cube currently sits on this one. The oracle uses
                     this to redirect users away from picking a covered base.

Tool effects:

* ``APPROACH(obj)`` — gripper.cell = obj.cell, z = HIGH (hover over the base).
* ``ALIGN_YAW(obj)`` — rotate gripper yaw to match obj.yaw.
* ``STACK(obj)`` — the held cube becomes ``stacked_on = obj`` and lands at obj.cell;
                   obj.top_of_stack flips to False; gripper releases (z → LOW).
* ``RELEASE`` — drop the held cube on the current cell (no stacking semantics).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import grid
from . import yaw as yawlib

CUBE_LABELS: Tuple[str, ...] = (
    "red_cube", "green_cube", "blue_cube", "yellow_cube", "purple_cube",
)

Z_BINS: Tuple[str, ...] = ("HIGH", "MID", "LOW")


@dataclass
class Cube:
    id: str
    label: str
    cell: str
    yaw: str
    is_held: bool = False
    stacked_on: Optional[str] = None
    top_of_stack: bool = True

    def to_record(self) -> Dict:
        return {
            "id": self.id,
            "label": self.label,
            "cell": self.cell,
            "yaw": self.yaw,
            "is_held": self.is_held,
            "stacked_on": self.stacked_on,
            "top_of_stack": self.top_of_stack,
        }


@dataclass
class Pose:
    cell: str
    yaw: str
    z: str

    def to_record(self) -> Dict:
        return {"cell": self.cell, "yaw": self.yaw, "z": self.z}


class EpisodeStacking:
    """Cube-stacking analogue of ``episode.Episode``.

    Mirrors that class's surface (``objects``, ``gripper_hist``, ``T``,
    ``apply_user_motion``, ``apply_tool``, ``get_obj``, ``intended_obj``,
    ``gripper_candidates``) so the data-generation loop in ``generate_dataset``
    can dispatch on env name without branching everywhere.
    """

    def __init__(self, rng: random.Random, episode_id: int, *, n_cubes: int = 3, prestack_p: float = 0.30):
        if not (2 <= n_cubes <= len(CUBE_LABELS)):
            raise ValueError("n_cubes must be in [2, len(CUBE_LABELS)]")
        self.rng = rng
        self.episode_id = episode_id
        self.T = int(rng.randint(10, 18))

        labels = list(CUBE_LABELS)
        rng.shuffle(labels)
        labels = labels[:n_cubes]

        # Lay all cubes on distinct cells (no overlap so stack semantics stay clean).
        cells_pool = list(grid.CELLS)
        rng.shuffle(cells_pool)

        objects: List[Cube] = []
        for i, label in enumerate(labels):
            objects.append(
                Cube(
                    id=f"c{i}",
                    label=label,
                    cell=cells_pool[i],
                    yaw=rng.choice(list(yawlib.YAW_BINS)),
                )
            )

        # One cube starts held.
        held = rng.choice(objects)
        held.is_held = True
        # Pre-existing stack: with prestack_p, designate a "base" and stack
        # another cube on top of it (so top_of_stack flips). The held cube cannot
        # be the base or the topper.
        on_table = [o for o in objects if not o.is_held]
        if len(on_table) >= 2 and rng.random() < float(prestack_p):
            base, topper = rng.sample(on_table, 2)
            topper.stacked_on = base.id
            topper.cell = base.cell
            base.top_of_stack = False

        self.objects = objects

        # Hidden ground truth: which uncovered base the user wants to stack on.
        valid_bases = [o for o in objects if not o.is_held and o.top_of_stack]
        # Fallback if every uncovered base got covered up; pick anything not held.
        if not valid_bases:
            valid_bases = [o for o in objects if not o.is_held]
        self.intended_obj_id = rng.choice(valid_bases).id

        # Gripper starts hovering somewhere on the grid.
        init_pose = Pose(
            cell=rng.choice(list(grid.CELLS)),
            yaw=rng.choice(list(yawlib.YAW_BINS)),
            z=rng.choice(("HIGH", "MID")),
        )
        self.gripper_hist: List[Pose] = [init_pose]
        while len(self.gripper_hist) < 6:
            self.apply_user_motion()

    # ------------------------------------------------------------------ helpers

    def get_obj(self, obj_id: str) -> Cube:
        for o in self.objects:
            if o.id == obj_id:
                return o
        raise KeyError(obj_id)

    def intended_obj(self) -> Cube:
        return self.get_obj(self.intended_obj_id)

    def held_obj(self) -> Optional[Cube]:
        return next((o for o in self.objects if o.is_held), None)

    def gripper_candidates(self, max_dist: int = 1) -> List[str]:
        """Bases that are not currently held and not covered, within ``max_dist``."""
        cell = self.gripper_hist[-1].cell
        out: List[str] = []
        for o in self.objects:
            if o.is_held or not o.top_of_stack:
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
        intended = self.intended_obj()

        if self.rng.random() < 0.8:
            next_cell = grid.step_toward(cur.cell, intended.cell)
        else:
            neigh = grid.neighbors(cur.cell)
            next_cell = self.rng.choice(neigh) if neigh else cur.cell

        # Cubes don't really require yaw alignment, but we keep ALIGN_YAW alive
        # for symmetry with YCB — so simulate occasional yaw drift.
        if cur.cell == intended.cell and self.rng.random() < 0.5:
            yn = yawlib.neighbors(intended.yaw)
            next_yaw = self.rng.choice([cur.yaw, yn[0], yn[1]])
        else:
            next_yaw = yawlib.move_toward(cur.yaw, intended.yaw, steps=1)

        # Z descends as we close in (the user is preparing to place).
        if next_cell == intended.cell:
            next_z = "MID" if cur.z == "HIGH" else ("LOW" if cur.z == "MID" else "LOW")
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
            set_gripper(cell=obj.cell, z="HIGH")
            return
        if tool == "ALIGN_YAW":
            obj = self.get_obj(args["obj"])
            set_gripper(yaw=obj.yaw)
            return
        if tool == "STACK":
            base = self.get_obj(args["obj"])
            held = self.held_obj()
            if held is None:
                # No-op: stacking without a held cube is undefined; let user-sim drive recovery.
                return
            held.is_held = False
            held.stacked_on = base.id
            held.cell = base.cell
            held.top_of_stack = True
            base.top_of_stack = False
            set_gripper(cell=base.cell, z="LOW")
            return
        if tool == "RELEASE":
            held = self.held_obj()
            if held is None:
                return
            held.is_held = False
            held.cell = cur.cell
            set_gripper(z="LOW")
            return
        raise ValueError(f"Unknown stacking tool: {tool}")
