"""
Compute UserPriors from a trial's gui_events.jsonl.

The shape is a strict superset of what user-study-prime/extract_data.py already
computes for per-trial reporting; we additionally compute rates per second of
active task time so the scripted user in `evaluation/scripted_user.py` can be
calibrated against them directly.

We treat *active task time* as the interval between the first and last command
in `gui_events.jsonl`. If fewer than two commands fired (e.g. the trial was a
no-op), all per-second rates collapse to 0 and the burst stats are zeroed.
"""

from __future__ import annotations

from typing import Any, Dict, List

from evaluation.scenarios.schema import UserPriors


# We only need to distinguish translation / rotation / gripper for the priors.
# `cartesian_velocity` events carry mode ∈ {translation, rotation}; gripper
# actions show up as separate `gripper` events (or `mode_change` to gripper +
# a finger toggle). The exact event names mirror the production code in
# kinova-isaac/.../gui_event_logger.py and are stable across all 160 trials
# we have on disk.


def _is_active_command(parsed: Dict[str, Any]) -> bool:
    if parsed.get("type") == "cartesian_velocity" and parsed.get("active"):
        return True
    if parsed.get("type") == "gripper":
        return True
    return False


def _command_mode(parsed: Dict[str, Any]) -> str:
    """Map a gui event to one of {"translation", "rotation", "gripper"}."""
    t = parsed.get("type", "")
    if t == "gripper":
        return "gripper"
    m = parsed.get("mode", "")
    if m in ("translation", "rotation"):
        return m
    return "translation"  # fallback; very rare


def _key_for_reversal(parsed: Dict[str, Any]) -> Any:
    return (parsed.get("mode", ""), parsed.get("axis", ""))


def compute_user_priors(gui_events: List[Dict[str, Any]]) -> UserPriors:
    """Derive UserPriors from the (already-parsed) gui_events list."""
    bursts_sec: List[float] = []
    mode_seconds = {"translation": 0.0, "rotation": 0.0, "gripper": 0.0}
    mode_switches = 0
    direction_reversals = 0

    active_start: Dict[Any, float] = {}
    last_command_mode: str | None = None
    last_direction_by_key: Dict[Any, int] = {}

    first_stamp: float | None = None
    last_stamp: float | None = None
    n_commands = 0

    for parsed in gui_events:
        t = parsed.get("type", "")
        stamp = float(parsed.get("stamp", 0.0))

        # Track the activity window from the first/last command.
        if _is_active_command(parsed):
            first_stamp = stamp if first_stamp is None else first_stamp
            last_stamp = stamp
            n_commands += 1

            cur_mode = _command_mode(parsed)
            if last_command_mode is not None and cur_mode != last_command_mode:
                mode_switches += 1
            last_command_mode = cur_mode

            if t == "cartesian_velocity":
                key = _key_for_reversal(parsed)
                direction = int(parsed.get("direction", 0))
                if key in last_direction_by_key and direction != 0:
                    if last_direction_by_key[key] != direction:
                        direction_reversals += 1
                if direction != 0:
                    last_direction_by_key[key] = direction
                active_start[key] = stamp

        elif t == "stop" and parsed.get("reason") == "release":
            key = _key_for_reversal(parsed)
            start = active_start.pop(key, None)
            if start is not None:
                dt = stamp - start
                if 0 < dt < 30:  # filter pathological bursts
                    bursts_sec.append(dt)
                    cur_mode = _command_mode(parsed)
                    mode_seconds[cur_mode if cur_mode in mode_seconds else "translation"] += dt

    total_active_sec = sum(bursts_sec)
    if total_active_sec > 0:
        trans_share = mode_seconds["translation"] / total_active_sec
        rot_share = mode_seconds["rotation"] / total_active_sec
        grip_share = mode_seconds["gripper"] / total_active_sec
    else:
        trans_share = rot_share = grip_share = 0.0

    window = (last_stamp - first_stamp) if (first_stamp is not None and last_stamp is not None) else 0.0
    if window > 0:
        ms_per_sec = mode_switches / window
        dr_per_sec = direction_reversals / window
    else:
        ms_per_sec = dr_per_sec = 0.0

    mean_burst = (sum(bursts_sec) / len(bursts_sec)) if bursts_sec else 0.0

    return UserPriors(
        translation_share=round(trans_share, 4),
        rotation_share=round(rot_share, 4),
        gripper_share=round(grip_share, 4),
        mean_active_burst_sec=round(mean_burst, 4),
        mode_switches_per_sec=round(ms_per_sec, 4),
        direction_reversals_per_sec=round(dr_per_sec, 4),
        total_active_time_sec=round(total_active_sec, 3),
        total_commands=n_commands,
    )


__all__ = ["compute_user_priors"]
