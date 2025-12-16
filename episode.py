from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import grid
import yaw as yawlib


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
LAST_OUTCOMES: Tuple[str, ...] = (
    "none",
    "stable_hover",
    "missed_contact",
    "collision",
    "grasp_success",
    "grasp_fail",
    "moved_object",
)


@dataclass
class Obj:
    id: str
    label: str
    cell: str
    yaw_bin: str
    is_held: bool = False

    def to_obs(self) -> Dict:
        return {
            "id": self.id,
            "label": self.label,
            "cell": self.cell,
            "yaw_bin": self.yaw_bin,
            "is_held": self.is_held,
        }


@dataclass
class Pose:
    cell: str
    yaw_bin: str
    z: str

    def to_obs(self) -> Dict:
        return {"cell": self.cell, "yaw_bin": self.yaw_bin, "z": self.z}


def _object_cell_counts(objects: Sequence[Obj]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for o in objects:
        if o.is_held:
            continue
        counts[o.cell] = counts.get(o.cell, 0) + 1
    return counts


class Episode:
    def __init__(self, rng, episode_id: int, n_obj: int, collision_p: float = 0.15):
        if not (2 <= n_obj <= len(OBJECT_LABELS)):
            raise ValueError("n_obj must be in [2, len(OBJECT_LABELS)]")
        self.rng = rng
        self.episode_id = episode_id
        self.T = int(rng.randint(8, 25))  # 8..25 inclusive

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
            yb = rng.choice(list(yawlib.YAW_BINS))
            objects.append(Obj(id=obj_id, label=label, cell=cell, yaw_bin=yb))

        self.objects = objects
        self.intended_obj_id: str = rng.choice([o.id for o in objects])

        init_pose = Pose(
            cell=rng.choice(list(grid.CELLS)),
            yaw_bin=rng.choice(list(yawlib.YAW_BINS)),
            z="HIGH",
        )
        self.gripper_hist: List[Pose] = [init_pose] * 6
        self.last_action_outcome: str = "none"

    def candidates(self) -> List[str]:
        return [o.id for o in self.objects if not o.is_held]

    def get_obj(self, obj_id: str) -> Obj:
        for o in self.objects:
            if o.id == obj_id:
                return o
        raise KeyError(obj_id)

    def intended_obj(self) -> Obj:
        return self.get_obj(self.intended_obj_id)

    def is_collision_cell(self, cell: str) -> bool:
        return _object_cell_counts(self.objects).get(cell, 0) >= 2

    def _push_gripper(self, pose: Pose) -> None:
        self.gripper_hist.append(pose)
        if len(self.gripper_hist) > 6:
            self.gripper_hist = self.gripper_hist[-6:]

    def apply_user_teleop_step(self) -> None:
        """
        Simulates a noisy intent-driven teleoperation update (cell/yaw/z).
        Only called on steps where the assistant is not directly moving the gripper.
        """
        cur = self.gripper_hist[-1]
        intended = self.intended_obj()

        # Cell motion
        if self.rng.random() < 0.75:
            next_cell = grid.step_toward(cur.cell, intended.cell)
        else:
            neigh = grid.neighbors(cur.cell)
            next_cell = self.rng.choice(neigh) if neigh else cur.cell

        # Yaw motion
        if self.rng.random() < 0.7:
            next_yaw = yawlib.move_toward(cur.yaw_bin, intended.yaw_bin, steps=1)
        else:
            next_yaw = self.rng.choice(list(yawlib.YAW_BINS))

        # Z motion (tends to go down when on intended cell)
        next_z = cur.z
        if next_cell == intended.cell:
            if cur.z == "HIGH":
                next_z = "MID" if self.rng.random() < 0.75 else "HIGH"
            elif cur.z == "MID":
                next_z = "LOW" if self.rng.random() < 0.7 else "MID"
            else:  # LOW
                next_z = "LOW" if self.rng.random() < 0.85 else "MID"
        else:
            if cur.z == "LOW":
                next_z = "MID" if self.rng.random() < 0.8 else "LOW"
            elif cur.z == "MID":
                next_z = "HIGH" if self.rng.random() < 0.6 else "MID"
            else:
                next_z = "HIGH"

        self._push_gripper(Pose(cell=next_cell, yaw_bin=next_yaw, z=next_z))

    def observe(self) -> Dict:
        return {
            "objects": [o.to_obs() for o in self.objects],
            "gripper_hist": [p.to_obs() for p in self.gripper_hist],
            "candidates": self.candidates(),
            "last_action_outcome": self.last_action_outcome,
        }

    def step_tool(self, tool_call: Dict) -> str:
        """
        Applies a tool call to the environment, updating state and returning outcome.
        """
        name = tool_call["tool_name"]
        args = tool_call["arguments"]
        cur = self.gripper_hist[-1]

        outcome = "none"

        def set_gripper(cell: Optional[str] = None, yaw_bin: Optional[str] = None, z: Optional[str] = None):
            self._push_gripper(
                Pose(
                    cell=cell if cell is not None else cur.cell,
                    yaw_bin=yaw_bin if yaw_bin is not None else cur.yaw_bin,
                    z=z if z is not None else cur.z,
                )
            )

        if name == "INTERACT":
            outcome = "none"
        elif name == "SELECT_TARGET":
            outcome = "none"
        elif name == "APPROACH":
            obj = self.get_obj(args["obj_id"])
            set_gripper(cell=obj.cell, z="HIGH")
            outcome = "stable_hover"
        elif name == "ALIGN_YAW":
            obj = self.get_obj(args["obj_id"])
            if self.rng.random() < 0.9:
                set_gripper(yaw_bin=obj.yaw_bin)
                outcome = "stable_hover"
            else:
                outcome = "missed_contact"
        elif name == "GRASP":
            obj = self.get_obj(args["obj_id"])
            same_cell = cur.cell == obj.cell
            yaw_match = cur.yaw_bin == obj.yaw_bin
            is_low = cur.z == "LOW"
            collision = self.is_collision_cell(obj.cell)

            p = 0.0
            if same_cell:
                p += 0.5
            if is_low:
                p += 0.3
            if yaw_match:
                p += 0.2
            if collision:
                p -= 0.2
            p = min(1.0, max(0.0, p))

            if self.rng.random() < p:
                obj.is_held = True
                outcome = "grasp_success"
            else:
                if not same_cell or not is_low:
                    outcome = "missed_contact"
                else:
                    outcome = "grasp_fail"
                    if collision and self.rng.random() < 0.2:
                        outcome = "moved_object"
        elif name == "RETRY_OR_ABORT":
            # Meta-action: leave state unchanged; outcome remains none.
            outcome = "none"
        else:
            raise ValueError(f"Unknown tool: {name}")

        self.last_action_outcome = outcome
        return outcome


def write_jsonl(path: str, records: Sequence[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
