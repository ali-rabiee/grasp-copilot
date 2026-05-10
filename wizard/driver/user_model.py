"""Simulated user that drives the gripper toward a (hidden) target.

This is the same low-bandwidth, noisy joystick model used in Package 3 so the
WoZ training distribution matches the real user-study distribution.

Per tick, the user emits one of:

* a translation step toward the target cell (with reversal/jitter probability),
* a rotation step toward the target yaw bin,
* a gripper-z change (rare),
* a no-op.

It also occasionally requests a mode switch, mirroring real-user behavior of
toggling between translate/rotate/gripper without any other reason.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Optional

from data_generator import grid as gridlib
from data_generator import yaw as yawlib

from ..env.schematic_env import SchematicEnv, USER_MODES, Z_BINS


@dataclass
class UserConfig:
    direction_reversal_p: float = 0.10  # take a step *away* from target
    drift_p: float = 0.10               # take a perpendicular/wrong step
    noop_p: float = 0.10                # press nothing this tick
    mode_switch_p: float = 0.05         # request a new control mode
    seed: Optional[int] = None


class SimulatedUser:
    """Stateful, target-conditioned, noisy joystick driver."""

    def __init__(self, cfg: UserConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

    # ------------------------------------------------------------- per-tick

    def step(self, env: SchematicEnv) -> Dict:
        """Return a command dict suitable for ``SchematicEnv.apply_user_command``.

        Side effect: may also request a mode change via ``env.set_user_mode``
        when the simulated user "presses" a mode-switch button.
        """
        if self.rng.random() < self.cfg.mode_switch_p:
            env.set_user_mode(self.rng.choice(USER_MODES))

        if self.rng.random() < self.cfg.noop_p:
            return {"mode": env.user_mode, "noop": True}

        target = next((o for o in env.objects if o.id == env.intended_obj_id), None)
        if target is None:
            return {"mode": env.user_mode, "noop": True}

        if env.user_mode == "translation":
            return self._translation_step(env, target.cell)
        if env.user_mode == "rotation":
            return self._rotation_step(env, target.yaw)
        return self._gripper_step(env)

    # ----------------------------------------------------------- internals

    def _translation_step(self, env: SchematicEnv, target_cell: str) -> Dict:
        cur = env.gripper.cell
        if cur == target_cell:
            return {"mode": "translation", "noop": True}

        ideal = gridlib.step_toward(cur, target_cell)
        nbrs = gridlib.neighbors(cur)

        if self.rng.random() < self.cfg.direction_reversal_p:
            away = [n for n in nbrs if n != ideal]
            chosen = self.rng.choice(away) if away else ideal
        elif self.rng.random() < self.cfg.drift_p:
            drift = [n for n in nbrs if n != ideal]
            chosen = self.rng.choice(drift) if drift else ideal
        else:
            chosen = ideal

        return {"mode": "translation", "step_cell": chosen}

    def _rotation_step(self, env: SchematicEnv, target_yaw: str) -> Dict:
        cur = env.gripper.yaw
        if cur == target_yaw:
            return {"mode": "rotation", "noop": True}

        ideal = yawlib.move_toward(cur, target_yaw, steps=1)
        if self.rng.random() < self.cfg.direction_reversal_p:
            wrong = yawlib.neighbors(cur)
            wrong = [w for w in wrong if w != ideal]
            chosen = self.rng.choice(wrong) if wrong else ideal
        else:
            chosen = ideal
        return {"mode": "rotation", "step_yaw": chosen}

    def _gripper_step(self, env: SchematicEnv) -> Dict:
        z = self.rng.choice(Z_BINS)
        return {"mode": "gripper", "z": z}

    # --------------------------------------------------- replies to INTERACT

    def reply_to_prompt(self, env: SchematicEnv, kind: str, choices: list[str]) -> str:
        """Generate a deterministic-with-noise reply to the wizard's INTERACT.

        Behavior is intent-aware (the simulated user "knows" its target):
          * CONFIRM → YES iff the prompt's first choice references the target.
          * QUESTION with object-label choices → pick the choice matching target label.
          * Fallback → first choice.
        """
        target = next((o for o in env.objects if o.id == env.intended_obj_id), None)
        if target is None:
            return choices[0] if choices else "1) OK"

        upper_choices = [c.upper() for c in choices]
        is_yesno = any("YES" in c for c in upper_choices) and any("NO" in c for c in upper_choices)

        if is_yesno:
            yes_idx = next(i for i, c in enumerate(upper_choices) if "YES" in c)
            no_idx = next(i for i, c in enumerate(upper_choices) if "NO" in c)
            mentions_target = target.label.lower() in (env.memory.get("last_prompt", {}).get("text", "")).lower()
            return choices[yes_idx if mentions_target else no_idx]

        for c in choices:
            if target.label.lower() in c.lower():
                return c

        return choices[0] if choices else "1) OK"
