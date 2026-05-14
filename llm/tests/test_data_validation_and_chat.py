import json
from pathlib import Path

from llm.data import (
    SYSTEM_PROMPT,
    convert_contract_to_qwen_chat_jsonl,
    convert_generator_jsonl_to_contract,
    validate_dataset_contract_jsonl,
)
from llm.rebalance_contract import rebalance_contract


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
                    "last_prompt": {"kind": "QUESTION", "text": "Which one?", "choices": ["1) mug"]},
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


def test_system_prompt_mentions_multi_env_tools():
    for tool in ("APPROACH", "ALIGN_YAW", "STACK", "GRAB", "POUR"):
        assert tool in SYSTEM_PROMPT
    assert '"amount": "SMALL"|"HALF"|"FULL"' in SYSTEM_PROMPT


def test_rebalance_repeats_task_action_tools(tmp_path: Path):
    contract = tmp_path / "contract.jsonl"
    rows = [
        {"id": "i", "instruction": "x", "input": "{}", "output": "{\"tool\":\"INTERACT\",\"args\":{\"kind\":\"CONFIRM\",\"text\":\"ok?\",\"choices\":[\"1) YES\",\"2) NO\"]}}"},
        {"id": "s", "instruction": "x", "input": "{}", "output": "{\"tool\":\"STACK\",\"args\":{\"obj\":\"o1\"}}"},
        {"id": "g", "instruction": "x", "input": "{}", "output": "{\"tool\":\"GRAB\",\"args\":{\"obj\":\"o0\"}}"},
        {"id": "p", "instruction": "x", "input": "{}", "output": "{\"tool\":\"POUR\",\"args\":{\"obj\":\"o2\",\"amount\":\"HALF\"}}"},
    ]
    contract.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "rebalanced.jsonl"

    stats = rebalance_contract(in_path=str(contract), out_path=str(out), seed=0, motion_repeat=3)
    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]

    assert stats["kept_motion"] == 3
    assert stats["kept_interact"] == 1
    assert stats["written"] == 10
    assert sum(1 for row in written if row["id"].startswith("s_m")) == 3
    assert sum(1 for row in written if row["id"].startswith("g_m")) == 3
    assert sum(1 for row in written if row["id"].startswith("p_m")) == 3
