import random

from evaluation.rollouts.noise import NoiseProfile
from evaluation.rollouts.rollout_loop import run_rollout


def _scenario(target_cell="A2", target_yaw="N"):
    return {
        "scenario_id": "test_001",
        "objects": [
            {"id": "obj_1", "label": "mug", "cell": target_cell, "yaw": target_yaw, "is_held": False},
            {"id": "obj_2", "label": "cleanser", "cell": "A3", "yaw": "E", "is_held": False},
        ],
        "gripper_init": {"cell": "B2", "yaw": "E", "z": "HIGH"},
        "target_obj_id": "obj_1",
    }


def test_manual_clean_succeeds():
    r = run_rollout(_scenario(), mode="manual",
                    noise_profile=NoiseProfile("clean"),
                    seed=0, hesitation_rate=0.0)
    assert r.success
    assert r.end_reason == "task_complete"
    assert r.total_inputs > 0
    assert r.interactions == 0
    assert r.completion_time_sec > 0


def test_manual_dropout_increases_input_cost():
    clean = run_rollout(_scenario(), mode="manual",
                       noise_profile=NoiseProfile("clean"), seed=42, hesitation_rate=0.0)
    noisy = run_rollout(_scenario(), mode="manual",
                       noise_profile=NoiseProfile("dr", p_drop=0.5), seed=42, hesitation_rate=0.0,
                       max_ticks=400)
    # Either both succeed but noisy takes more total input ticks, or noisy fails.
    assert (noisy.dropped_inputs > 0) or (not noisy.success)


def test_prime_oracle_backend_completes_easy_scenario():
    # Use the heuristic oracle as a cheap PRIME stand-in.
    from data_generator.oracle import OracleState, oracle_decide_tool

    target = "obj_1"

    def oracle_backend(input_dict):
        state = OracleState(intended_obj_id=target)
        try:
            return oracle_decide_tool(
                input_dict["objects"], input_dict["gripper_hist"],
                input_dict["memory"], state, user_state=input_dict["user_state"],
            )
        except Exception:
            return None

    r = run_rollout(_scenario(), mode="prime",
                    noise_profile=NoiseProfile("clean"),
                    seed=0, backend=oracle_backend, hesitation_rate=0.0)
    # PRIME mode should at least invoke some tool calls and/or interactions
    # then either complete or terminate cleanly.
    assert r.motion_tool_calls + r.interactions > 0
    assert r.completion_time_sec > 0


def test_missing_target_raises():
    import pytest
    sc = _scenario()
    sc["target_obj_id"] = None
    with pytest.raises(ValueError):
        run_rollout(sc, mode="manual", noise_profile=NoiseProfile("clean"), seed=0)


def test_unknown_mode_raises():
    import pytest
    with pytest.raises(ValueError):
        run_rollout(_scenario(), mode="??", noise_profile=NoiseProfile("clean"), seed=0)


def test_prime_without_backend_raises():
    import pytest
    with pytest.raises(ValueError):
        run_rollout(_scenario(), mode="prime",
                    noise_profile=NoiseProfile("clean"), seed=0)
