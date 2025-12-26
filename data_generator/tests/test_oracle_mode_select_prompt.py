from data_generator.oracle import OracleState, oracle_decide_tool


def test_oracle_awaiting_mode_select_always_prompts():
    objects = [{"id": "o0", "label": "mug", "cell": "A1", "yaw": "N", "is_held": False}]
    gripper_hist = [{"cell": "A1", "yaw": "N", "z": "HIGH"}] * 6
    memory = {"candidates": ["o0"], "past_dialogs": [], "n_interactions": 1, "last_tool_calls": [], "excluded_obj_ids": []}
    state = OracleState(intended_obj_id="o0", awaiting_mode_select=True)

    tool = oracle_decide_tool(objects, gripper_hist, memory, state, user_state={"mode": "translation"})
    assert tool["tool"] == "INTERACT"
    assert "approaching" in tool["args"]["text"].lower()
    assert "aligning" in tool["args"]["text"].lower()
    assert tool["args"]["choices"] == ["1) APPROACH", "2) ALIGN_YAW"]


