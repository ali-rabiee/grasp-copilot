from data_generator.oracle import OracleState, oracle_decide_tool


def test_oracle_offers_followup_align_after_approach():
    objects = [{"id": "o0", "label": "mug", "cell": "A1", "yaw": "E", "is_held": False}]
    # After approach: cell matches object, yaw does not.
    gripper_hist = [{"cell": "A1", "yaw": "N", "z": "HIGH"}] * 6
    memory = {
        "candidates": ["o0"],
        "excluded_obj_ids": [],
        "past_dialogs": [],
        "n_interactions": 2,
        "last_tool_calls": ["APPROACH"],
        "last_action": {"tool": "APPROACH", "obj": "o0"},
    }
    state = OracleState(intended_obj_id="o0")
    tool = oracle_decide_tool(objects, gripper_hist, memory, state, user_state={"mode": "translation"})
    assert tool["tool"] == "INTERACT"
    assert tool["args"]["kind"] == "CONFIRM"
    assert "align yaw" in tool["args"]["text"].lower()


