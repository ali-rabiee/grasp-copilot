"""
Tests for the per-scenario priors-calibrated ScriptedUser.

These verify the load-bearing claim of Path 1: when constructed from a
trial's real `user_priors`, the scripted user emits ~total_commands commands
per trial rather than ~minimal_path_length.
"""

import random

from data_generator.episode import Episode
from evaluation.rollouts.scripted_user import ScriptedUser, _minimal_path_length
from evaluation.rollouts.noise import NoiseProfile
from evaluation.rollouts.rollout_loop import run_rollout


def _scenario(target_cell="A2", target_yaw="N",
              grip_cell="B2", grip_yaw="E", grip_z="HIGH",
              total_commands=48, mean_active_burst_sec=0.35,
              direction_reversals_per_sec=0.04):
    return {
        "scenario_id": "test",
        "objects": [
            {"id": "obj_1", "label": "mug", "cell": target_cell, "yaw": target_yaw, "is_held": False},
            {"id": "obj_2", "label": "cleanser", "cell": "A3", "yaw": "E", "is_held": False},
        ],
        "gripper_init": {"cell": grip_cell, "yaw": grip_yaw, "z": grip_z},
        "target_obj_id": "obj_1",
        "user_priors": {
            "total_commands": total_commands,
            "mean_active_burst_sec": mean_active_burst_sec,
            "direction_reversals_per_sec": direction_reversals_per_sec,
            "translation_share": 0.7,
            "rotation_share": 0.2,
            "gripper_share": 0.1,
        },
        "difficulty": "easy",
    }


def test_minimal_path_length_basics():
    s = _scenario(target_cell="B2", target_yaw="E", grip_cell="B2", grip_yaw="E", grip_z="HIGH")
    # Same cell, same yaw, descend 2, close: 3 steps total.
    assert _minimal_path_length(s) == 3
    s2 = _scenario(target_cell="A2", target_yaw="N", grip_cell="C3", grip_yaw="S", grip_z="HIGH")
    # 3 cell moves + 4 yaw + 2 z descend + 1 close = 10
    assert _minimal_path_length(s2) == 10


def test_from_priors_calibrates_bursts_per_step():
    s = _scenario(total_commands=48)
    min_path = _minimal_path_length(s)
    user = ScriptedUser.from_priors(
        target_obj_id="obj_1", rng=random.Random(0),
        priors=s["user_priors"], scenario=s,
    )
    # 48 / 6 = 8 bursts per step (for default Easy-like scenario)
    assert user.bursts_per_step == max(1, -(-48 // min_path))   # ceil
    assert user.tick_dt_sec == 0.35
    # 0.04 reversals/sec × 0.35 sec/burst = 0.014 hesitation rate
    assert abs(user.hesitation_rate - 0.014) < 1e-6


def test_from_priors_with_zero_or_missing_falls_back_to_defaults():
    s = _scenario()
    user = ScriptedUser.from_priors(
        target_obj_id="obj_1", rng=random.Random(0),
        priors={}, scenario=s,
    )
    assert user.bursts_per_step == 1
    assert user.tick_dt_sec == 0.3
    assert user.hesitation_rate == 0.0


def test_priors_user_emits_many_bursts_per_step():
    """Each cell-move logical step should emit `bursts_per_step` bursts."""
    s = _scenario(target_cell="A2", target_yaw="E", grip_cell="A1", grip_yaw="E", grip_z="LOW",
                  total_commands=10, mean_active_burst_sec=0.1, direction_reversals_per_sec=0.0)
    # Min path: 1 cell + 0 yaw + 0 z + 1 close = 2. bursts_per_step = 10/2 = 5.
    ep = Episode.from_scenario(s, rng=random.Random(0))
    user = ScriptedUser.from_priors(
        target_obj_id="obj_1", rng=random.Random(0),
        priors=s["user_priors"], scenario=s,
    )
    assert user.bursts_per_step == 5

    # Emit until the cell changes; first 4 bursts should be _advance_sim=False.
    advances = []
    for _ in range(5):
        cmd = user.next_command(ep)
        advances.append(cmd["_advance_sim"])
        if cmd.get("_advance_sim"):
            ep.step_user_command(axis=cmd["axis"], direction=cmd["direction"], mode=cmd["mode"])
    assert advances == [False, False, False, False, True]
    assert ep.gripper_hist[-1].cell == "A2"


def test_priors_rollout_total_inputs_scales_up():
    """With priors-derived bursts_per_step, total_inputs should approach total_commands."""
    s = _scenario(total_commands=48, mean_active_burst_sec=0.35, direction_reversals_per_sec=0.0)
    r_with = run_rollout(s, mode="manual", noise_profile=NoiseProfile("clean"),
                        seed=0, max_ticks=2000)
    r_without = run_rollout(s, mode="manual", noise_profile=NoiseProfile("clean"),
                           seed=0, max_ticks=2000, use_per_scenario_priors=False)
    assert r_with.success
    assert r_without.success
    # The priors-calibrated rollout should emit substantially more inputs
    # than the default-1-burst-per-step one.
    assert r_with.total_inputs >= 3 * r_without.total_inputs


def test_priors_rollout_completion_time_scales_up():
    s = _scenario(total_commands=48, mean_active_burst_sec=0.35, direction_reversals_per_sec=0.0)
    r_with = run_rollout(s, mode="manual", noise_profile=NoiseProfile("clean"),
                        seed=0, max_ticks=2000)
    r_without = run_rollout(s, mode="manual", noise_profile=NoiseProfile("clean"),
                           seed=0, max_ticks=2000, use_per_scenario_priors=False)
    assert r_with.completion_time_sec > r_without.completion_time_sec
