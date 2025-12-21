import pytest

from data_generator.oracle import validate_tool_call


def test_validate_tool_call_rejects_more_than_5_choices():
    tool = {
        "tool": "INTERACT",
        "args": {
            "kind": "QUESTION",
            "text": "pick",
            "choices": ["1) a", "2) b", "3) c", "4) d", "5) e", "6) f"],
        },
    }
    with pytest.raises(ValueError):
        validate_tool_call(tool)


