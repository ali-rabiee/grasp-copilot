from data_generator.oracle import OracleState, oracle_decide_tool


def test_oracle_when_no_candidates_after_exclusion_asks_anything_else():
    # Simulate the "None of them" path that excluded all candidates.
    objects = [
        {"id": "o0", "label": "mug", "cell": "A1", "yaw": "N", "is_held": False},
        {"id": "o1", "label": "sugar_box", "cell": "A1", "yaw": "N", "is_held": False},
    ]
    gripper_hist = [{"cell": "A1", "yaw": "N", "z": "HIGH"}] * 6
    memory = {"candidates": [], "past_dialogs": [], "n_interactions": 1, "last_tool_calls": [], "excluded_obj_ids": ["o0", "o1"]}
    state = OracleState(intended_obj_id="o0", awaiting_choice=True)

    tool = oracle_decide_tool(objects, gripper_hist, memory, state, user_state={"mode": "translation"})
    assert tool["tool"] == "INTERACT"
    assert "anything else" in tool["args"]["text"].lower()


