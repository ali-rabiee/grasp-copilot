from evaluation.scenarios.schema import (
    LAYOUT_SOURCE_STATE_SNAPSHOT,
    TARGET_SOURCE_TOOL_CALL,
    TARGET_SOURCE_UNLABELED,
    GripperInit,
    ObjectInit,
    Scenario,
    TrialSource,
    UserPriors,
    load_scenarios,
    write_scenarios,
)


def _scenario(target=None, source=TARGET_SOURCE_UNLABELED):
    return Scenario(
        scenario_id="test_001",
        source=TrialSource(mode="assistive", subject="s1", difficulty="easy",
                           trial_id="trial_xxx", trial_dir="PRIME_LOGS/assistive/s1/easy/trial_xxx"),
        objects=[
            ObjectInit(id="obj_1", label="mug", raw_label="mug", cell="A2", yaw="N"),
            ObjectInit(id="obj_2", label="cleanser", raw_label="cleanser", cell="A3", yaw="E"),
        ],
        gripper_init=GripperInit(cell="B2", yaw="E", z="HIGH"),
        target_obj_id=target,
        target_label_source=source,
        layout_source=LAYOUT_SOURCE_STATE_SNAPSHOT,
        user_priors=UserPriors(),
        difficulty="easy",
        plausible_target_ids=["obj_1", "obj_2"],
    )


def test_validate_accepts_well_formed_scenario():
    s = _scenario()
    assert s.validate() == []


def test_validate_catches_target_not_in_objects():
    s = _scenario(target="obj_99", source=TARGET_SOURCE_TOOL_CALL)
    problems = s.validate()
    assert any("obj_99" in p for p in problems)


def test_validate_catches_inconsistent_target_source():
    # target is None but source claims tool_call → inconsistent
    s = _scenario(target=None, source=TARGET_SOURCE_TOOL_CALL)
    problems = s.validate()
    assert any("inconsistent" in p for p in problems)


def test_roundtrip_jsonl(tmp_path):
    src = [_scenario(target="obj_1", source=TARGET_SOURCE_TOOL_CALL),
           _scenario(target=None, source=TARGET_SOURCE_UNLABELED)]
    src[1].scenario_id = "test_002"
    path = tmp_path / "scenarios.jsonl"
    write_scenarios(path, src)
    loaded = load_scenarios(path)
    assert len(loaded) == 2
    assert loaded[0].target_obj_id == "obj_1"
    assert loaded[1].target_obj_id is None
    assert loaded[0].objects[0].label == "mug"
