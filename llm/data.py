from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, NoReturn, Optional, Sequence, Tuple

from .utils import json_loads_strict


SYSTEM_PROMPT = "You are a helpful assistant. Output ONLY valid JSON."


@dataclass(frozen=True, slots=True)
class DatasetExample:
    id: str
    instruction: str
    input: str
    output: str  # must parse as JSON


def _fail(path: str, line_no: int, msg: str) -> NoReturn:
    raise ValueError(f"{path}:{line_no}: {msg}")


def iter_jsonl(path: str) -> Iterator[Tuple[int, Dict]]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj: Dict
            try:
                obj = json.loads(line)
            except Exception as e:
                _fail(path, i, f"Invalid JSON: {e}")
            if not isinstance(obj, dict):
                _fail(path, i, "Expected JSON object per line")
            yield i, obj


def validate_dataset_contract_jsonl(path: str) -> None:
    """
    Dataset contract:
      - id: string
      - instruction: string
      - input: string (may be empty)
      - output: string (must parse as JSON)
    """
    for line_no, obj in iter_jsonl(path):
        missing = {k for k in ("id", "instruction", "input", "output") if k not in obj}
        if missing:
            _fail(path, line_no, f"Missing keys: {sorted(missing)}")
        if not isinstance(obj["id"], str):
            _fail(path, line_no, "id must be a string")
        if not isinstance(obj["instruction"], str):
            _fail(path, line_no, "instruction must be a string")
        if not isinstance(obj["input"], str):
            _fail(path, line_no, "input must be a string")
        if not isinstance(obj["output"], str):
            _fail(path, line_no, "output must be a string")
        try:
            json_loads_strict(obj["output"])
        except Exception as e:
            _fail(path, line_no, f"output must be valid JSON string: {e}")


def dataset_contract_to_qwen_chat_messages(ex: DatasetExample) -> Dict:
    user = ex.instruction
    if ex.input:
        user = f"{user}\n\nInput:\n{ex.input}"
    return {
        "id": ex.id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": ex.output},
        ],
    }


def load_dataset_contract(path: str) -> List[DatasetExample]:
    validate_dataset_contract_jsonl(path)
    out: List[DatasetExample] = []
    for _, obj in iter_jsonl(path):
        out.append(
            DatasetExample(
                id=obj["id"],
                instruction=obj["instruction"],
                input=obj["input"],
                output=obj["output"],
            )
        )
    return out


def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def convert_generator_jsonl_to_contract(
    generator_path: str,
    out_path: str,
    instruction: Optional[str] = None,
) -> None:
    """
    Thin adapter to reuse the existing generator output.

    Expected generator record keys:
      - episode_id, objects, gripper_hist, memory, user_state, target_tool_call
    Produces dataset-contract JSONL with output as a JSON string.
    """
    if instruction is None:
        instruction = (
            "Given the robot observation and dialog context, infer the user's intent and "
            "emit exactly one tool call. Output ONLY the tool call JSON with keys tool and args. "
            "If the tool is INTERACT, you must output at most 5 choices total."
        )

    rows: List[Dict] = []
    for line_no, obj in iter_jsonl(generator_path):
        for k in ("episode_id", "objects", "gripper_hist", "memory", "user_state", "target_tool_call"):
            if k not in obj:
                _fail(generator_path, line_no, f"Missing key: {k}")
        ex_id = f"{obj['episode_id']}_{line_no}"
        input_blob = {
            "objects": obj["objects"],
            "gripper_hist": obj["gripper_hist"],
            "memory": obj["memory"],
            "user_state": obj["user_state"],
        }
        output_obj = obj["target_tool_call"]
        if not isinstance(output_obj, dict):
            _fail(generator_path, line_no, "target_tool_call must be an object")
        output_str = json.dumps(output_obj, ensure_ascii=False, separators=(",", ":"))
        json_loads_strict(output_str)  # sanity
        rows.append(
            {
                "id": ex_id,
                "instruction": instruction,
                "input": json.dumps(input_blob, ensure_ascii=False),
                "output": output_str,
            }
        )

    write_jsonl(out_path, rows)
    validate_dataset_contract_jsonl(out_path)


def convert_contract_to_qwen_chat_jsonl(contract_path: str, out_path: str) -> None:
    examples = load_dataset_contract(contract_path)
    rows = [dataset_contract_to_qwen_chat_messages(ex) for ex in examples]
    write_jsonl(out_path, rows)
