import json
from pathlib import Path

from llm.data import (
    convert_contract_to_qwen_chat_jsonl,
    convert_generator_jsonl_to_contract,
    validate_dataset_contract_jsonl,
)


def test_validate_and_convert_contract_to_chat(tmp_path: Path):
    contract = tmp_path / "contract.jsonl"
    contract.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "0",
                        "instruction": "Return a tool call.",
                        "input": "{\"obs\":{}}",
                        "output": "{\"tool\":\"INTERACT\",\"args\":{\"kind\":\"SUGGESTION\",\"text\":\"ok\",\"choices\":[\"1) YES\",\"2) NO\"]}}",
                    }
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    validate_dataset_contract_jsonl(str(contract))

    chat = tmp_path / "chat.jsonl"
    convert_contract_to_qwen_chat_jsonl(str(contract), str(chat))
    line = chat.read_text(encoding="utf-8").strip()
    obj = json.loads(line)
    assert obj["messages"][0]["role"] == "system"
    assert obj["messages"][1]["role"] == "user"
    assert obj["messages"][2]["role"] == "assistant"


def test_convert_generator_jsonl_to_contract(tmp_path: Path):
    gen = tmp_path / "gen.jsonl"
    gen.write_text(
        json.dumps(
            {
                "episode_id": 1,
                "objects": [{"id": "o0", "label": "mug", "cell": "A1", "yaw": "N", "is_held": False}],
                "gripper_hist": [
                    {"cell": "A1", "yaw": "N", "z": "MID"},
                    {"cell": "A1", "yaw": "N", "z": "MID"},
                    {"cell": "A1", "yaw": "N", "z": "MID"},
                    {"cell": "A1", "yaw": "N", "z": "MID"},
                    {"cell": "A1", "yaw": "N", "z": "MID"},
                    {"cell": "A1", "yaw": "N", "z": "MID"},
                ],
                "memory": {
                    "past_dialogs": [{"role": "user", "content": "mug"}],
                    "candidates": ["o0"],
                    "n_interactions": 1,
                    "last_tool_calls": ["INTERACT"],
                },
                "user_state": {"mode": "translation"},
                "target_tool_call": {"tool": "APPROACH", "args": {"obj": "o0"}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "contract.jsonl"
    convert_generator_jsonl_to_contract(str(gen), str(out))
    validate_dataset_contract_jsonl(str(out))
