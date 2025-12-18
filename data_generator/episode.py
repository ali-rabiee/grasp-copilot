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


def write_jsonl(path: str, records: Sequence[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

