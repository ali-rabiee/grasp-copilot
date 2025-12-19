from data_generator.oracle import OracleState, oracle_decide_tool


def test_oracle_followup_align_yes_executes_align_instead_of_reprompt():
    objects = [{"id": "o0", "label": "mug", "cell": "A1", "yaw": "E", "is_held": False}]
    gripper_hist = [{"cell": "A1", "yaw": "N", "z": "HIGH"}] * 6

    # Simulate that we just asked the follow-up and the user clicked YES:
    memory = {
        "candidates": ["o0"],
        "excluded_obj_ids": [],
        "past_dialogs": [{"role": "assistant", "content": "Do you want me to also align yaw to the mug?"}, {"role": "user", "content": "YES"}],
        "n_interactions": 3,
        "last_tool_calls": ["INTERACT"],
        "last_action": {"tool": "APPROACH", "obj": "o0"},
    }
    state = OracleState(intended_obj_id="o0")
    state.pending_action_obj_id = "o0"
    state.pending_mode = "ALIGN_YAW"
    state.selected_obj_id = "o0"
    state.awaiting_confirmation = False

    tool = oracle_decide_tool(objects, gripper_hist, memory, state, user_state={"mode": "rotation"})
    assert tool["tool"] == "ALIGN_YAW"
    assert tool["args"]["obj"] == "o0"


