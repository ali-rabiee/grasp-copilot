from data_generator.oracle import OracleState, oracle_decide_tool


def test_oracle_ambiguity_triggers_interact_question():
    obs = {
        "objects": [
            {"id": "o0", "label": "mug", "cell": "A1", "yaw_bin": "N", "is_held": False},
            {"id": "o1", "label": "sugar_box", "cell": "A1", "yaw_bin": "E", "is_held": False},
            {"id": "o2", "label": "tuna_fish_can", "cell": "C3", "yaw_bin": "S", "is_held": False},
        ],
        "gripper_hist": [
            {"cell": "A2", "yaw_bin": "N", "z": "HIGH"},
            {"cell": "A2", "yaw_bin": "N", "z": "HIGH"},
            {"cell": "A2", "yaw_bin": "N", "z": "HIGH"},
            {"cell": "A2", "yaw_bin": "N", "z": "HIGH"},
            {"cell": "A2", "yaw_bin": "N", "z": "HIGH"},
            {"cell": "A1", "yaw_bin": "N", "z": "HIGH"},
        ],
        "candidates": ["o0", "o1", "o2"],
        "last_action_outcome": "none",
    }
    dialog = []
    state = OracleState()
    tool = oracle_decide_tool(obs, dialog, state)
    assert tool["tool_name"] == "INTERACT"
    assert tool["arguments"]["type"] == "question"
    assert set(tool["arguments"]["choices"]) == {"mug", "sugar_box"}

