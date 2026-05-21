import random

from data_generator.episode import Episode
from evaluation.rollouts.scripted_user import ScriptedUser, _yaw_step_direction, _cell_translation_step


def _minimal_scenario(target_cell="A2", target_yaw="N",
                       grip_cell="B2", grip_yaw="E", grip_z="HIGH"):
    return {
        "scenario_id": "test",
        "objects": [
            {"id": "obj_1", "label": "mug", "cell": target_cell, "yaw": target_yaw, "is_held": False},
            {"id": "obj_2", "label": "cleanser", "cell": "A3", "yaw": "E", "is_held": False},
        ],
        "gripper_init": {"cell": grip_cell, "yaw": grip_yaw, "z": grip_z},
        "target_obj_id": "obj_1",
    }


def test_yaw_step_direction_picks_shorter_arc():
    # N → NE is +1 step CW; N → NW is -1 step CCW.
    assert _yaw_step_direction("N", "NE") == +1
    assert _yaw_step_direction("N", "NW") == -1
    # Equidistant on 8-ring: tie-break clockwise (matches yawlib.move_toward).
    assert _yaw_step_direction("N", "S") == +1


def test_cell_translation_step_prefers_row_then_col():
    # cur=B2 → target A2: row first (decrease row by 1).
    assert _cell_translation_step("B2", "A2") == ("y", -1)
    # cur=A1 → target A3: row matches; advance column.
    assert _cell_translation_step("A1", "A3") == ("x", +1)
    # Already on target.
    assert _cell_translation_step("B2", "B2") is None


def test_user_drives_to_target():
    sc = _minimal_scenario(target_cell="A2", target_yaw="N", grip_cell="B2", grip_yaw="E", grip_z="HIGH")
    ep = Episode.from_scenario(sc, rng=random.Random(0))
    user = ScriptedUser(target_obj_id="obj_1", rng=random.Random(0), hesitation_rate=0.0)

    steps = 0
    while not user.is_done(ep) and steps < 50:
        cmd = user.next_command(ep)
        assert cmd is not None
        # Strip rollout-only metadata before handing to the simulator.
        ep.step_user_command(axis=cmd["axis"], direction=cmd["direction"], mode=cmd["mode"])
        steps += 1

    assert user.is_done(ep)
    assert steps < 50, "user should terminate well under the cap"


def test_user_is_done_requires_all_four_conditions():
    sc = _minimal_scenario(target_cell="B2", target_yaw="E", grip_cell="B2", grip_yaw="E", grip_z="HIGH")
    ep = Episode.from_scenario(sc, rng=random.Random(0))
    user = ScriptedUser(target_obj_id="obj_1", rng=random.Random(0))

    # Cell + yaw match but z=HIGH, gripper not closed → not done.
    assert not user.is_done(ep)
    ep.step_user_command(axis="z", direction=-1, mode="translation")  # → MID
    ep.step_user_command(axis="z", direction=-1, mode="translation")  # → LOW
    assert not user.is_done(ep)
    ep.step_user_command(axis="", direction=0, mode="gripper")
    assert user.is_done(ep)


def test_answer_prompt_picks_target_label_when_present():
    sc = _minimal_scenario()
    ep = Episode.from_scenario(sc, rng=random.Random(0))
    user = ScriptedUser(target_obj_id="obj_1", rng=random.Random(0))

    call = {"tool": "INTERACT", "args": {
        "kind": "QUESTION",
        "text": "Which object?",
        "choices": ["1) cleanser", "2) mug", "3) None of them"],
    }}
    idx, hit = user.answer_prompt(call, ep)
    assert idx == 1
    assert hit is True


def test_answer_prompt_falls_back_to_none_when_target_absent():
    sc = _minimal_scenario()
    ep = Episode.from_scenario(sc, rng=random.Random(0))
    user = ScriptedUser(target_obj_id="obj_1", rng=random.Random(0))

    call = {"tool": "INTERACT", "args": {
        "kind": "QUESTION",
        "text": "Which object?",
        "choices": ["1) cleanser", "2) coffee_can", "3) None of them"],
    }}
    idx, hit = user.answer_prompt(call, ep)
    assert idx == 2
    assert hit is False


def test_tick_cost_includes_mode_switch():
    user = ScriptedUser(target_obj_id="obj_1", rng=random.Random(0),
                        tick_dt_sec=0.3, mode_switch_cost_sec=0.6)
    c1 = user.tick_cost_sec({"axis": "x", "direction": 1, "mode": "translation"})
    c2 = user.tick_cost_sec({"axis": "x", "direction": 1, "mode": "translation"})
    c3 = user.tick_cost_sec({"axis": "yaw", "direction": 1, "mode": "rotation"})
    assert abs(c1 - 0.3) < 1e-9   # first command, no switch
    assert abs(c2 - 0.3) < 1e-9   # same mode
    assert abs(c3 - 0.9) < 1e-9   # switch translation → rotation (0.3 + 0.6)
