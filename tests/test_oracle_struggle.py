from oracle import OracleState, oracle_decide_tool


def test_oracle_struggle_triggers_offer_takeover_on_oscillation():
    obs = {
        "objects": [{"id": "o0", "label": "mug", "cell": "B2", "yaw_bin": "N", "is_held": False}],
        "gripper_hist": [
            {"cell": "B2", "yaw_bin": "N", "z": "HIGH"},
            {"cell": "B2", "yaw_bin": "E", "z": "HIGH"},
            {"cell": "B2", "yaw_bin": "N", "z": "HIGH"},
            {"cell": "B2", "yaw_bin": "E", "z": "HIGH"},
            {"cell": "B2", "yaw_bin": "N", "z": "HIGH"},
            {"cell": "B2", "yaw_bin": "E", "z": "HIGH"},
        ],
        "candidates": ["o0"],
        "last_action_outcome": "none",
    }
    state = OracleState()
    tool = oracle_decide_tool(obs, [], state)
    assert tool["tool_name"] == "INTERACT"
    assert tool["arguments"]["type"] == "offer_takeover"


def test_oracle_struggle_triggers_offer_takeover_on_failures():
    obs = {
        "objects": [{"id": "o0", "label": "mug", "cell": "B2", "yaw_bin": "N", "is_held": False}],
        "gripper_hist": [
            {"cell": "A1", "yaw_bin": "N", "z": "HIGH"},
            {"cell": "A1", "yaw_bin": "N", "z": "HIGH"},
            {"cell": "A1", "yaw_bin": "N", "z": "HIGH"},
            {"cell": "A1", "yaw_bin": "N", "z": "HIGH"},
            {"cell": "A1", "yaw_bin": "N", "z": "HIGH"},
            {"cell": "A1", "yaw_bin": "N", "z": "HIGH"},
        ],
        "candidates": ["o0"],
        "last_action_outcome": "grasp_fail",
    }
    state = OracleState(outcomes=["missed_contact", "none", "grasp_fail"])
    tool = oracle_decide_tool(obs, [], state)
    assert tool["tool_name"] == "INTERACT"
    assert tool["arguments"]["type"] == "offer_takeover"


def test_oracle_takeover_yes_suppresses_repeat_offer_takeover():
    # Construct a case that would normally trigger oscillation-based struggle.
    obs = {
        "objects": [{"id": "o0", "label": "mug", "cell": "B2", "yaw_bin": "N", "is_held": False}],
        "gripper_hist": [
            {"cell": "B2", "yaw_bin": "N", "z": "HIGH"},
            {"cell": "B2", "yaw_bin": "E", "z": "HIGH"},
            {"cell": "B2", "yaw_bin": "N", "z": "HIGH"},
            {"cell": "B2", "yaw_bin": "E", "z": "HIGH"},
            {"cell": "B2", "yaw_bin": "N", "z": "HIGH"},
            {"cell": "B2", "yaw_bin": "E", "z": "HIGH"},
        ],
        "candidates": ["o0"],
        "last_action_outcome": "none",
    }
    dialog = [
        {"role": "assistant", "content": "Want me to align yaw / take over?"},
        {"role": "user", "content": "yes please"},
    ]
    state = OracleState()
    tool = oracle_decide_tool(obs, dialog, state)
    # After user accepts takeover, oracle should proceed with action routing,
    # not immediately re-offer takeover.
    assert tool["tool_name"] != "INTERACT" or tool["arguments"].get("type") != "offer_takeover"
    assert tool["tool_name"] == "SELECT_TARGET"

