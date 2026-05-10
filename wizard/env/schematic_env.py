"""Sim-free schematic environment for the PRIME wizard.

The wizard never sees IsaacSim. This module provides a faithful symbolic
counterpart: a 3x3 grid workspace, a set of objects, a gripper with discrete
cell/yaw/z state, and the four memory components PRIME's LLM consumes
(``candidates``, ``past_dialogs``, ``last_action``, ``excluded_obj_ids``).

Three task families are supported and produce structurally different scenes
so a single trained model sees a unified ask-or-act distribution:

* ``reach_to_grasp_ycb`` — clutter of YCB-like labels, no held object.
* ``cube_stacking``     — small set of colored cubes; one is pre-held.
* ``pouring``           — a pitcher (held or pickable) + a target cup.

The output blob matches ``grasp-copilot/data/runs/*/grasp_gen.jsonl`` line
schema verbatim, except ``target_tool_call`` is filled in by the wizard
(see ``wizard.io.writer``) instead of the heuristic oracle.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from data_generator import grid as gridlib
from data_generator import yaw as yawlib

ENVS = ("reach_to_grasp_ycb", "cube_stacking", "pouring")
Z_BINS: Tuple[str, ...] = ("LOW", "MID", "HIGH")
USER_MODES: Tuple[str, ...] = ("translation", "rotation", "gripper")

YCB_LABELS: Tuple[str, ...] = (
    "mug", "master_chef_can", "gelatin_box", "mustard_bottle", "bleach_cleanser",
    "potted_meat_can", "tuna_fish_can", "tomato_soup_can", "pudding_box",
    "sugar_box", "banana", "cracker_box", "apple", "orange",
)
CUBE_LABELS: Tuple[str, ...] = ("red_cube", "green_cube", "blue_cube", "yellow_cube")
POURING_LABELS: Tuple[str, ...] = ("pitcher", "cup_a", "cup_b")


@dataclass
class GripperState:
    cell: str
    yaw: str
    z: str = "LOW"


@dataclass
class ObjectState:
    id: str
    label: str
    cell: str
    yaw: str
    is_held: bool = False


@dataclass
class EnvConfig:
    env_name: str = "reach_to_grasp_ycb"
    n_objects_min: int = 2
    n_objects_max: int = 6
    candidate_max_dist: int = 2
    history_len: int = 6
    seed: Optional[int] = None


def _new_id(idx: int) -> str:
    return f"o{idx}"


def _sample_unique_cells(rng: random.Random, n: int) -> List[str]:
    cells = list(gridlib.CELLS)
    rng.shuffle(cells)
    out: List[str] = []
    while len(out) < n:
        out.append(cells[len(out) % len(cells)])
    return out[:n]


def sample_scene(cfg: EnvConfig, rng: random.Random) -> Tuple[List[ObjectState], GripperState, str, str]:
    """Return (objects, gripper, intended_obj_id, initial_user_mode).

    ``intended_obj_id`` is the simulated user's secret target — written to the
    episode-level metadata for offline analysis but **never** exposed in the
    per-tick blob the wizard sees.
    """
    if cfg.env_name == "cube_stacking":
        labels_pool = CUBE_LABELS
        n = rng.randint(max(2, cfg.n_objects_min), min(4, cfg.n_objects_max))
    elif cfg.env_name == "pouring":
        labels_pool = POURING_LABELS
        n = min(3, max(2, cfg.n_objects_max))
    else:
        labels_pool = YCB_LABELS
        n = rng.randint(cfg.n_objects_min, cfg.n_objects_max)

    cells = _sample_unique_cells(rng, n)
    labels = list(rng.sample(labels_pool, k=min(n, len(labels_pool))))
    while len(labels) < n:
        labels.append(rng.choice(labels_pool))

    objects: List[ObjectState] = []
    for i in range(n):
        objects.append(
            ObjectState(
                id=_new_id(i),
                label=labels[i],
                cell=cells[i],
                yaw=rng.choice(yawlib.YAW_BINS),
                is_held=False,
            )
        )

    if cfg.env_name == "cube_stacking" and objects:
        objects[0].is_held = True

    pickable = [o for o in objects if not o.is_held]
    intended = rng.choice(pickable).id if pickable else objects[0].id

    g_cell = rng.choice([c for c in gridlib.CELLS if c not in {o.cell for o in objects}] or list(gridlib.CELLS))
    gripper = GripperState(cell=g_cell, yaw=rng.choice(yawlib.YAW_BINS), z="MID")

    return objects, gripper, intended, "translation"


class SchematicEnv:
    """Discrete 3x3 grid world. No physics, no rendering — pure symbolic state."""

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.episode_idx = -1
        self.objects: List[ObjectState] = []
        self.gripper = GripperState(cell="B2", yaw="N", z="MID")
        self.gripper_hist: Deque[Dict[str, str]] = deque(maxlen=cfg.history_len)
        self.user_mode: str = "translation"
        self.intended_obj_id: str = ""
        self.memory: Dict = self._fresh_memory()
        self.tick: int = 0

    @staticmethod
    def _fresh_memory() -> Dict:
        return {
            "n_interactions": 0,
            "past_dialogs": [],
            "candidates": [],
            "last_tool_calls": [],
            "excluded_obj_ids": [],
            "last_action": {},
            "last_prompt": {},
        }

    # ------------------------------------------------------------------ reset

    def reset(self) -> None:
        self.episode_idx += 1
        objs, grip, intended, mode = sample_scene(self.cfg, self.rng)
        self.objects = objs
        self.gripper = grip
        self.intended_obj_id = intended
        self.user_mode = mode
        self.gripper_hist.clear()
        for _ in range(self.cfg.history_len):
            self.gripper_hist.append(self._gripper_record())
        self.memory = self._fresh_memory()
        self.memory["candidates"] = self._compute_candidates()
        self.tick = 0

    def _gripper_record(self) -> Dict[str, str]:
        return {"cell": self.gripper.cell, "yaw": self.gripper.yaw, "z": self.gripper.z}

    # -------------------------------------------------------------- candidates

    def _compute_candidates(self) -> List[str]:
        excluded = set(self.memory.get("excluded_obj_ids") or [])
        cur = self.gripper.cell
        cands: List[str] = []
        for o in self.objects:
            if o.is_held or o.id in excluded:
                continue
            if gridlib.manhattan(cur, o.cell) <= self.cfg.candidate_max_dist:
                cands.append(o.id)
        return cands

    def _refresh_candidates(self) -> None:
        self.memory["candidates"] = self._compute_candidates()

    # ----------------------------------------------------------- gripper apply

    def apply_user_command(self, cmd: Dict) -> None:
        """Apply a noisy user command to the gripper.

        Commands are produced by ``driver.user_model.SimulatedUser``. Schema:

            {"mode": "translation", "step_cell": "B2"}      # next cell
            {"mode": "rotation",    "step_yaw":  "NE"}      # next yaw bin
            {"mode": "gripper",     "z": "LOW"}             # change z bin
            {"mode": "<m>",         "noop": True}

        Mode switches are handled by ``set_user_mode``, not here.
        """
        mode = cmd.get("mode", self.user_mode)
        if cmd.get("noop"):
            self.gripper_hist.append(self._gripper_record())
            self.tick += 1
            return
        if mode == "translation" and "step_cell" in cmd:
            self.gripper.cell = cmd["step_cell"]
        elif mode == "rotation" and "step_yaw" in cmd:
            self.gripper.yaw = cmd["step_yaw"]
        elif mode == "gripper" and "z" in cmd:
            if cmd["z"] in Z_BINS:
                self.gripper.z = cmd["z"]
        self.gripper_hist.append(self._gripper_record())
        self._refresh_candidates()
        self.tick += 1

    def set_user_mode(self, mode: str) -> None:
        if mode in USER_MODES:
            self.user_mode = mode

    # ------------------------------------------------------------- skill apply

    def apply_execution_skill(self, tool: str, obj_id: str) -> str:
        """Execute APPROACH or ALIGN_YAW deterministically. Returns outcome label."""
        target = next((o for o in self.objects if o.id == obj_id), None)
        if target is None:
            return "fail_target_missing"
        if tool == "APPROACH":
            self.gripper.cell = target.cell
            self.gripper.z = "LOW"
            outcome = "success"
        elif tool == "ALIGN_YAW":
            self.gripper.yaw = target.yaw
            outcome = "success"
        else:
            return "fail_unknown_tool"
        self.gripper_hist.append(self._gripper_record())
        self.memory["last_action"] = {"tool": tool, "obj": obj_id, "outcome": outcome}
        last_calls = list(self.memory.get("last_tool_calls") or [])
        last_calls.append(tool)
        self.memory["last_tool_calls"] = last_calls[-3:]
        self._refresh_candidates()
        self.tick += 1
        return outcome

    def apply_interaction(self, kind: str, text: str, choices: List[str], reply: str) -> None:
        """Record an INTERACT turn and let the user reply update memory."""
        self.memory["n_interactions"] = int(self.memory.get("n_interactions", 0)) + 1
        past = list(self.memory.get("past_dialogs") or [])
        past.append({"prompt": text, "kind": kind, "choices": list(choices), "reply": reply})
        self.memory["past_dialogs"] = past[-6:]
        self.memory["last_prompt"] = {"kind": kind, "text": text, "choices": list(choices)}
        last_calls = list(self.memory.get("last_tool_calls") or [])
        last_calls.append("INTERACT")
        self.memory["last_tool_calls"] = last_calls[-3:]

    def exclude_obj(self, obj_id: str) -> None:
        ex = set(self.memory.get("excluded_obj_ids") or [])
        ex.add(obj_id)
        self.memory["excluded_obj_ids"] = sorted(ex)
        self._refresh_candidates()

    # ----------------------------------------------------------------- export

    def public_blob(self) -> Dict:
        """The exact symbolic blob the wizard (and the deployed LLM) sees.

        Excludes ``intended_obj_id`` and any other ground-truth fields.
        """
        return {
            "objects": [
                {"id": o.id, "label": o.label, "cell": o.cell, "yaw": o.yaw, "is_held": o.is_held}
                for o in self.objects
            ],
            "gripper_hist": list(self.gripper_hist),
            "memory": {
                "n_interactions": int(self.memory.get("n_interactions", 0)),
                "past_dialogs": list(self.memory.get("past_dialogs") or []),
                "candidates": list(self.memory.get("candidates") or []),
                "last_tool_calls": list(self.memory.get("last_tool_calls") or []),
                "excluded_obj_ids": list(self.memory.get("excluded_obj_ids") or []),
                "last_action": dict(self.memory.get("last_action") or {}),
                "last_prompt": dict(self.memory.get("last_prompt") or {}),
            },
            "user_state": {"mode": self.user_mode},
        }
