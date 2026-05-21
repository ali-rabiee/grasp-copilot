"""
Deterministic target-aware scripted user for noise-robustness rollouts.

Per plan §5.4, the policy is intentionally simple — reviewers will scrutinize
complex learned user models. The policy decides one "logical step" per call:

  1. If gripper cell ≠ target cell → translation (closest axis first)
  2. Else if gripper yaw ≠ target yaw → rotation
  3. Else if gripper z ≠ LOW         → translation z down
  4. Else if not gripper_closed      → gripper toggle (closes the grasp)
  5. Else                            → done

The novel piece (plan Path 1) is the **per-scenario calibration**: real users
emit ~10 short bursts of joystick / head-array activity per logical step,
not one. So `next_command` can be configured to emit `bursts_per_step`
commands per logical step, with only the terminal burst actually advancing
the simulator's gripper. This brings the scripted user's emit count from
~5 per Easy trial up to the ~48 seen in real PRIME_LOGS data, without
changing the simulator semantics.

Calibration sources:

  Knob                       Default     Per-scenario source (from UserPriors)
  ─────────────────────────  ──────────  ────────────────────────────────────────
  bursts_per_step            1           ceil(total_commands / minimal_path_len)
  tick_dt_sec                0.3 s       mean_active_burst_sec
  hesitation_rate            0.05        direction_reversals_per_sec × tick_dt
  mode_switch_cost_sec       0.6 s       fixed (still global)

When the priors-derived knobs are missing or zero, the defaults above are
used — so unit tests that don't pass priors keep their old behaviour.

For INTERACT prompts from PRIME, `answer_prompt` returns the choice index
whose label/option corresponds to the target. If no choice maps to the
target, it returns the index of "None of them" if present, else 0; either
way the per-rollout result records `target_filtered_out = True` for
transparency.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from data_generator import grid, yaw as yawlib
from data_generator.episode import Episode


# ── helpers ─────────────────────────────────────────────────────────────────


def _yaw_step_direction(cur: str, target: str) -> int:
    """+1 means 'move clockwise on the 8-bin yaw ring will reduce distance'.

    Mirrors `yawlib.move_toward`'s clockwise/counter-clockwise tie-break
    (clockwise wins when equidistant).
    """
    n = len(yawlib.YAW_BINS)
    ci = yawlib.YAW_BINS.index(cur)
    ti = yawlib.YAW_BINS.index(target)
    cw = (ti - ci) % n
    ccw = (ci - ti) % n
    return +1 if cw <= ccw else -1


def _cell_translation_step(cur_cell: str, tgt_cell: str) -> Optional[Tuple[str, int]]:
    """Return one (axis, direction) reducing the Manhattan distance, or None.

    Row moves come first (matches `grid.step_toward`'s tie-break).
    """
    c = grid.Cell.from_label(cur_cell)
    t = grid.Cell.from_label(tgt_cell)
    if c.r != t.r:
        return ("y", +1 if t.r > c.r else -1)
    if c.c != t.c:
        return ("x", +1 if t.c > c.c else -1)
    return None


def _minimal_path_length(scenario) -> int:
    """Logical-step count from initial gripper pose to a grasp completion.

    Used by `from_priors` to compute the expected number of joystick bursts
    per logical step. Returns at least 1 to avoid division-by-zero.
    """
    objs = getattr(scenario, "objects", None) or scenario.get("objects", [])
    target_id = getattr(scenario, "target_obj_id", None) or scenario.get("target_obj_id")
    target = next((o for o in objs if (getattr(o, "id", None) or o.get("id")) == target_id), None)
    if target is None:
        return 1

    g = getattr(scenario, "gripper_init", None) or scenario.get("gripper_init", {})
    g_cell = getattr(g, "cell", None) or g.get("cell")
    g_yaw = getattr(g, "yaw", None) or g.get("yaw")
    g_z = getattr(g, "z", None) or g.get("z") or "HIGH"

    t_cell = getattr(target, "cell", None) or target.get("cell")
    t_yaw = getattr(target, "yaw", None) or target.get("yaw")

    cell_steps = grid.manhattan(g_cell, t_cell) if g_cell and t_cell else 0
    yaw_steps = (
        yawlib.cyclic_distance_steps(g_yaw, t_yaw) if g_yaw and t_yaw else 0
    )
    # z descent: HIGH → MID → LOW is 2 steps when starting at HIGH.
    z_steps = {"HIGH": 2, "MID": 1, "LOW": 0}.get(g_z, 2)
    gripper_close = 1
    return max(1, cell_steps + yaw_steps + z_steps + gripper_close)


# ── policy ──────────────────────────────────────────────────────────────────


@dataclass
class ScriptedUser:
    target_obj_id: str
    rng: random.Random
    hesitation_rate: float = 0.05
    mode_switch_cost_sec: float = 0.6
    tick_dt_sec: float = 0.3
    bursts_per_step: int = 1

    _last_cmd: Optional[Dict[str, Any]] = None
    _last_mode: Optional[str] = None
    _current_logical_cmd: Optional[Dict[str, Any]] = field(default=None, init=False)
    _bursts_emitted_this_step: int = field(default=0, init=False)

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def from_priors(
        cls,
        target_obj_id: str,
        rng: random.Random,
        priors: Any,
        scenario: Any,
        *,
        min_bursts_per_step: int = 1,
        max_bursts_per_step: int = 50,
        mode_switch_cost_sec: float = 0.6,
    ) -> "ScriptedUser":
        """Build a ScriptedUser calibrated to one scenario's per-trial priors.

        `priors` must expose:
            total_commands              int  (≥ 0)
            mean_active_burst_sec       float (s)
            direction_reversals_per_sec float (1/s)

        Either a dataclass `UserPriors` or a plain dict with those keys works.
        Missing / zero fields fall back to the class defaults.
        """

        def _g(name, default):
            if priors is None:
                return default
            v = getattr(priors, name, None)
            if v is None and isinstance(priors, dict):
                v = priors.get(name)
            try:
                return float(v) if v is not None else default
            except (TypeError, ValueError):
                return default

        total_cmds = _g("total_commands", 0.0)
        burst_sec = _g("mean_active_burst_sec", 0.0)
        rev_per_sec = _g("direction_reversals_per_sec", 0.0)

        min_path = _minimal_path_length(scenario)

        if total_cmds > 0:
            bursts_per_step = int(math.ceil(total_cmds / max(1, min_path)))
            bursts_per_step = max(min_bursts_per_step, min(max_bursts_per_step, bursts_per_step))
        else:
            bursts_per_step = min_bursts_per_step

        tick_dt = burst_sec if burst_sec > 0 else 0.3
        # Reversals per second × per-burst seconds → per-burst probability.
        hesitation = max(0.0, min(0.5, rev_per_sec * tick_dt))

        return cls(
            target_obj_id=target_obj_id,
            rng=rng,
            hesitation_rate=hesitation,
            mode_switch_cost_sec=mode_switch_cost_sec,
            tick_dt_sec=tick_dt,
            bursts_per_step=bursts_per_step,
        )

    # ── action selection ────────────────────────────────────────────────────

    def next_command(self, episode: Episode) -> Optional[Dict[str, Any]]:
        """Return one user command, or None if the task is already done.

        Output shape::

            {"axis": str, "direction": int, "mode": str,
             "_advance_sim": bool   # True only on the terminal burst of a logical step}

        The rollout loop should pass commands with ``_advance_sim=True`` to the
        simulator's ``step_user_command``; intermediate "tracking bursts" are
        counted toward `total_inputs` and the clock but do not change state.
        """
        if self.is_done(episode):
            return None

        # If we're mid-step, keep emitting the same logical command until
        # the burst quota is filled.
        if self._current_logical_cmd is not None and self._bursts_emitted_this_step < self.bursts_per_step:
            return self._emit_next_burst()

        # Start a fresh logical step.
        cmd = self._choose_command(episode)
        if cmd is None:
            return None
        self._current_logical_cmd = cmd
        self._bursts_emitted_this_step = 0
        return self._emit_next_burst()

    def _emit_next_burst(self) -> Dict[str, Any]:
        """Yield one burst from the current logical step.

        Gripper-mode commands are atomic (always 1 burst; sim advances immediately).
        For translation/rotation, the LAST burst of the step is the one that
        advances the simulator.
        """
        base = dict(self._current_logical_cmd)
        if base.get("mode") == "gripper":
            burst = dict(base, _advance_sim=True)
            self._bursts_emitted_this_step = self.bursts_per_step  # consume the step
        else:
            self._bursts_emitted_this_step += 1
            is_terminal = self._bursts_emitted_this_step >= self.bursts_per_step
            burst = dict(base, _advance_sim=is_terminal)

            # Hesitation: with low probability, flip direction on this burst.
            # The flipped burst doesn't advance the simulator regardless of
            # terminality — real users back off then resume.
            if (
                self.hesitation_rate > 0
                and self._last_cmd is not None
                and self.rng.random() < self.hesitation_rate
            ):
                burst["direction"] = -int(burst.get("direction", 1))
                burst["_advance_sim"] = False

        self._last_cmd = burst
        return burst

    def _choose_command(self, episode: Episode) -> Optional[Dict[str, Any]]:
        target = episode.get_obj(self.target_obj_id)
        cur = episode.gripper_hist[-1]

        # 1) Cell mismatch → translation
        cell_step = _cell_translation_step(cur.cell, target.cell)
        if cell_step is not None:
            axis, direction = cell_step
            return {"axis": axis, "direction": direction, "mode": "translation"}

        # 2) Yaw mismatch → rotation
        if cur.yaw != target.yaw:
            direction = _yaw_step_direction(cur.yaw, target.yaw)
            return {"axis": "yaw", "direction": direction, "mode": "rotation"}

        # 3) Need to descend → translation/z, downward
        if cur.z != "LOW":
            return {"axis": "z", "direction": -1, "mode": "translation"}

        # 4) Close gripper
        if not getattr(episode, "gripper_closed", False):
            return {"axis": "", "direction": 0, "mode": "gripper"}

        return None

    # ── prompt answering ────────────────────────────────────────────────────

    def answer_prompt(self, interact_call: Dict[str, Any], episode: Episode) -> Tuple[int, bool]:
        """Return (choice_index, target_in_options).

        `target_in_options` lets the rollout caller record whether PRIME's
        choice list actually included the target — useful diagnostic when
        running under selection noise.
        """
        args = interact_call.get("args", {}) or {}
        choices: List[str] = list(args.get("choices") or [])
        if not choices:
            return (0, False)

        target = episode.get_obj(self.target_obj_id)
        target_label = (target.label or "").lower()

        # Try to match by object label first (handles QUESTION/CONFIRM/SUGGESTION
        # prompts that list candidate objects).
        for i, choice in enumerate(choices):
            if target_label and target_label in choice.lower():
                return (i, True)

        # Common YES/NO / "anything_else" / "mode_select" branches: prefer YES /
        # affirmative answers because the user wants the assistance to continue.
        for i, choice in enumerate(choices):
            text = choice.lower()
            if "yes" in text or "ok" in text or "confirm" in text:
                return (i, False)

        # Fall back to "None of them" if present (signals PRIME to re-query
        # rather than silently picking a wrong object).
        for i, choice in enumerate(choices):
            if "none of them" in choice.lower() or "none" == choice.strip().lower():
                return (i, False)

        return (0, False)

    # ── termination ─────────────────────────────────────────────────────────

    def is_done(self, episode: Episode) -> bool:
        try:
            target = episode.get_obj(self.target_obj_id)
        except KeyError:
            return False
        cur = episode.gripper_hist[-1]
        return (
            cur.cell == target.cell
            and cur.yaw == target.yaw
            and cur.z == "LOW"
            and getattr(episode, "gripper_closed", False)
        )

    # ── clock book-keeping ──────────────────────────────────────────────────

    def tick_cost_sec(self, cmd: Dict[str, Any]) -> float:
        """Wall-clock cost to emit this command, including mode-switch overhead."""
        cost = self.tick_dt_sec
        cur_mode = cmd.get("mode")
        if self._last_mode is not None and cur_mode != self._last_mode:
            cost += self.mode_switch_cost_sec
        self._last_mode = cur_mode
        return cost


__all__ = ["ScriptedUser"]
