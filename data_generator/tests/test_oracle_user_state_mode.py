from data_generator.oracle import OracleState, oracle_decide_tool


def test_oracle_translation_mode_uses_approach_intent_gate():
    objects = [
        {"id": "o0", "label": "mug", "cell": "A1", "yaw": "N", "is_held": False},
        {"id": "o1", "label": "sugar_box", "cell": "A2", "yaw": "E", "is_held": False},
        {"id": "o2", "label": "tuna_fish_can", "cell": "A2", "yaw": "S", "is_held": False},
    ]
    gripper_hist = [{"cell": "A2", "yaw": "N", "z": "MID"}] * 6
    memory = {"candidates": ["o1", "o2"], "past_dialogs": [], "n_interactions": 0, "last_tool_calls": [], "excluded_obj_ids": []}
    state = OracleState(intended_obj_id="o1")
    tool = oracle_decide_tool(objects, gripper_hist, memory, state, user_state={"mode": "translation"})
    assert tool["tool"] == "INTERACT"
    assert tool["args"]["kind"] == "QUESTION"
    assert "approaching" in tool["args"]["text"]


def test_oracle_rotation_mode_prefers_align_intent_gate_text():
    objects = [
        {"id": "o0", "label": "mug", "cell": "A1", "yaw": "N", "is_held": False},
        {"id": "o1", "label": "sugar_box", "cell": "A2", "yaw": "E", "is_held": False},
        {"id": "o2", "label": "tuna_fish_can", "cell": "A2", "yaw": "S", "is_held": False},
    ]
    gripper_hist = [{"cell": "A2", "yaw": "N", "z": "MID"}] * 6
    memory = {"candidates": ["o1", "o2"], "past_dialogs": [], "n_interactions": 0, "last_tool_calls": [], "excluded_obj_ids": []}
    state = OracleState(intended_obj_id="o1")
    tool = oracle_decide_tool(objects, gripper_hist, memory, state, user_state={"mode": "rotation"})
    assert tool["tool"] == "INTERACT"
    assert tool["args"]["kind"] == "QUESTION"
    assert "align yaw" in tool["args"]["text"].lower()


