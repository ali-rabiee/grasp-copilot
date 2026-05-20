import json

from evaluation.scenarios.adapters import (
    GRIPPER_HIST_LEN,
    scenario_to_contract_row,
    scenario_to_input_dict,
    write_scenarios_as_contract_jsonl,
)
from evaluation.scenarios.schema import (
    LAYOUT_SOURCE_STATE_SNAPSHOT,
    TARGET_SOURCE_TOOL_CALL,
    TARGET_SOURCE_UNLABELED,
    GripperInit,
    ObjectInit,
    Scenario,
    TrialSource,
    UserPriors,
)


def _scenario(target="obj_1"):
    return Scenario(
        scenario_id="test_001",
        source=TrialSource(mode="assistive", subject="s1", difficulty="easy",
                           trial_id="t0", trial_dir="PRIME_LOGS/assistive/s1/easy/t0"),
        objects=[
            ObjectInit(id="obj_1", label="mug", raw_label="mug", cell="B2", yaw="N"),
            ObjectInit(id="obj_2", label="cleanser", raw_label="cleanser", cell="A3", yaw="E"),
        ],
        gripper_init=GripperInit(cell="B2", yaw="E", z="HIGH"),
        target_obj_id=target,
        target_label_source=TARGET_SOURCE_TOOL_CALL if target else TARGET_SOURCE_UNLABELED,
        layout_source=LAYOUT_SOURCE_STATE_SNAPSHOT,
        user_priors=UserPriors(),
        difficulty="easy",
        plausible_target_ids=["obj_1", "obj_2"],
    )


def test_input_dict_has_six_pose_history():
    inp = scenario_to_input_dict(_scenario())
    assert len(inp["gripper_hist"]) == GRIPPER_HIST_LEN
    # All six poses are identical at t=0 — signals "no motion yet."
    first = inp["gripper_hist"][0]
    for pose in inp["gripper_hist"][1:]:
        assert pose == first
    assert first == {"cell": "B2", "yaw": "E", "z": "HIGH"}


def test_input_dict_objects_drop_raw_label():
    inp = scenario_to_input_dict(_scenario())
    keys = set(inp["objects"][0].keys())
    assert keys == {"id", "label", "cell", "yaw", "is_held"}


def test_input_dict_candidates_computed_from_layout():
    inp = scenario_to_input_dict(_scenario())
    # obj_1 is at the gripper cell B2 → manhattan 0; obj_2 at A3 → manhattan 2.
    # With default max_dist=1, only obj_1 should be a candidate.
    assert inp["memory"]["candidates"] == ["obj_1"]


def test_input_dict_candidate_radius_override():
    inp = scenario_to_input_dict(_scenario(), candidate_max_dist=3)
    assert set(inp["memory"]["candidates"]) == {"obj_1", "obj_2"}


def test_input_dict_memory_is_empty_at_t0():
    inp = scenario_to_input_dict(_scenario())
    m = inp["memory"]
    assert m["n_interactions"] == 0
    assert m["past_dialogs"] == []
    assert m["last_tool_calls"] == []
    assert m["excluded_obj_ids"] == []
    assert m["last_action"] == {}
    assert "last_prompt" not in m   # only present after first INTERACT


def test_contract_row_shape():
    row = scenario_to_contract_row(_scenario())
    assert set(row.keys()) == {"id", "instruction", "input", "output"}
    # input/output are JSON-encoded strings.
    inp = json.loads(row["input"])
    assert "objects" in inp and "gripper_hist" in inp and "memory" in inp and "user_state" in inp
    assert json.loads(row["output"]) == {}


def test_contract_row_with_oracle_output():
    fake_output = {"tool": "APPROACH", "args": {"obj": "obj_1"}}
    row = scenario_to_contract_row(_scenario(), output_tool_call=fake_output)
    assert json.loads(row["output"]) == fake_output


def test_bulk_write_respects_skip_unlabeled(tmp_path):
    out = tmp_path / "contract.jsonl"
    scenarios = [_scenario(target="obj_1"), _scenario(target=None)]
    scenarios[1].scenario_id = "test_002"

    n = write_scenarios_as_contract_jsonl(scenarios, out, skip_unlabeled=True)
    assert n == 1
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    assert rows[0]["id"] == "test_001"


def test_bulk_write_keeps_unlabeled_by_default(tmp_path):
    out = tmp_path / "contract.jsonl"
    scenarios = [_scenario(target="obj_1"), _scenario(target=None)]
    scenarios[1].scenario_id = "test_002"
    n = write_scenarios_as_contract_jsonl(scenarios, out)
    assert n == 2
