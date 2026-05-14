from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .data import convert_contract_to_qwen_chat_jsonl, iter_jsonl, validate_dataset_contract_jsonl, write_jsonl
from .rebalance_contract import rebalance_contract
from .utils import json_loads_strict


DEFAULT_GENERATORS = (
    "data/woz_reach_to_grasp_ycb/grasp_gen.jsonl",
    "data/woz_cube_stacking/grasp_gen.jsonl",
    "data/woz_pouring/grasp_gen.jsonl",
)

DEFAULT_INSTRUCTION = (
    "Given the robot observation and dialog context, infer the user's intent and "
    "emit exactly one tool call. Output ONLY the tool call JSON with keys tool and args. "
    "If the tool is INTERACT, you must output at most 5 choices total."
)


def _source_name(path: str) -> str:
    name = Path(path).parent.name
    return name[4:] if name.startswith("woz_") else name


def _tool_from_row(row: Dict[str, Any]) -> str:
    output = json_loads_strict(str(row["output"]))
    if not isinstance(output, dict) or not isinstance(output.get("tool"), str):
        raise ValueError(f"Invalid output tool in row {row.get('id')!r}")
    return str(output["tool"])


def _convert_generator_rows(
    generator_path: str,
    *,
    source: str,
    instruction: str,
    max_past_dialogs: int,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    required = {"episode_id", "objects", "gripper_hist", "memory", "user_state", "target_tool_call"}

    for line_no, obj in iter_jsonl(generator_path):
        missing = required.difference(obj)
        if missing:
            raise ValueError(f"{generator_path}:{line_no}: Missing keys: {sorted(missing)}")

        mem = obj["memory"]
        if isinstance(mem, dict) and isinstance(mem.get("past_dialogs"), list) and int(max_past_dialogs) > 0:
            mem = dict(mem)
            mem["past_dialogs"] = list(mem.get("past_dialogs") or [])[-int(max_past_dialogs) :]

        output_obj = obj["target_tool_call"]
        if not isinstance(output_obj, dict):
            raise ValueError(f"{generator_path}:{line_no}: target_tool_call must be an object")

        input_blob = {
            "objects": obj["objects"],
            "gripper_hist": obj["gripper_hist"],
            "memory": mem,
            "user_state": obj["user_state"],
        }
        output_str = json.dumps(output_obj, ensure_ascii=False, separators=(",", ":"))
        json_loads_strict(output_str)

        rows.append(
            {
                "id": f"{source}_{obj['episode_id']}_{line_no}",
                "instruction": instruction,
                "input": json.dumps(input_blob, ensure_ascii=False, separators=(",", ":")),
                "output": output_str,
            }
        )

    return rows


def _split_by_source(
    rows_by_source: Dict[str, List[Dict[str, str]]],
    *,
    valid_fraction: float,
    seed: int,
) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    rng = random.Random(int(seed))
    train_rows: List[Dict[str, str]] = []
    valid_rows: List[Dict[str, str]] = []

    for source, rows in sorted(rows_by_source.items()):
        source_rows = list(rows)
        rng.shuffle(source_rows)
        n = len(source_rows)
        if n < 2:
            raise ValueError(f"Need at least two rows for source {source!r}; got {n}")
        n_valid = max(1, int(round(n * float(valid_fraction))))
        n_valid = min(n_valid, n - 1)
        valid_rows.extend(source_rows[:n_valid])
        train_rows.extend(source_rows[n_valid:])

    rng.shuffle(train_rows)
    rng.shuffle(valid_rows)
    return train_rows, valid_rows


def _counts(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(_tool_from_row(row) for row in rows))


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Prepare merged WOZ phase-2 train/valid contract files.")
    ap.add_argument("--generator_jsonl", action="append", default=None, help="WOZ generator JSONL. Can be repeated.")
    ap.add_argument("--out_dir", type=str, default="data/woz_phase2")
    ap.add_argument("--valid_fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_past_dialogs", type=int, default=12)
    ap.add_argument("--motion_repeat", type=int, default=2)
    ap.add_argument("--interact_keep_prob", type=float, default=1.0)
    ap.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    args = ap.parse_args(argv)

    generator_paths = list(args.generator_jsonl or DEFAULT_GENERATORS)
    rows_by_source: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for generator_path in generator_paths:
        source = _source_name(generator_path)
        rows_by_source[source].extend(
            _convert_generator_rows(
                generator_path,
                source=source,
                instruction=str(args.instruction),
                max_past_dialogs=int(args.max_past_dialogs),
            )
        )

    train_rows, valid_rows = _split_by_source(
        rows_by_source,
        valid_fraction=float(args.valid_fraction),
        seed=int(args.seed),
    )
    all_rows = [row for rows in rows_by_source.values() for row in rows]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_contract = out_dir / "llm_contract_all.jsonl"
    train_contract = out_dir / "llm_contract_train.jsonl"
    valid_contract = out_dir / "llm_contract_valid.jsonl"
    train_rebalanced = out_dir / "llm_contract_train_rebalanced.jsonl"

    write_jsonl(str(all_contract), all_rows)
    write_jsonl(str(train_contract), train_rows)
    write_jsonl(str(valid_contract), valid_rows)
    for path in (all_contract, train_contract, valid_contract):
        validate_dataset_contract_jsonl(str(path))

    stats = rebalance_contract(
        in_path=str(train_contract),
        out_path=str(train_rebalanced),
        seed=int(args.seed),
        motion_repeat=int(args.motion_repeat),
        interact_keep_prob=float(args.interact_keep_prob),
    )
    validate_dataset_contract_jsonl(str(train_rebalanced))

    convert_contract_to_qwen_chat_jsonl(str(train_contract), str(out_dir / "llm_chat_train.jsonl"))
    convert_contract_to_qwen_chat_jsonl(str(valid_contract), str(out_dir / "llm_chat_valid.jsonl"))
    convert_contract_to_qwen_chat_jsonl(str(train_rebalanced), str(out_dir / "llm_chat_train_rebalanced.jsonl"))

    summary = {
        "generators": generator_paths,
        "seed": int(args.seed),
        "valid_fraction": float(args.valid_fraction),
        "rows": {
            "all": len(all_rows),
            "train": len(train_rows),
            "valid": len(valid_rows),
            "train_rebalanced": int(stats["written"]),
        },
        "tool_distribution": {
            "all": _counts(all_rows),
            "train": _counts(train_rows),
            "valid": _counts(valid_rows),
            "train_rebalanced": _counts(row for _, row in iter_jsonl(str(train_rebalanced))),
        },
        "rebalance": stats,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
