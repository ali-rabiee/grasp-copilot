from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

from llm.data import convert_contract_to_qwen_chat_jsonl, validate_dataset_contract_jsonl, write_jsonl


ENV_DIRS = {
    "reach_to_grasp_ycb": "reach_to_grasp",
    "cube_stacking": "stacking",
    "pouring": "pouring",
}


def _load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _tool(row: Mapping) -> str:
    return str(json.loads(str(row["output"]))["tool"])


def _env_from_id(row_id: str, prefix: str) -> str:
    stem = row_id.removeprefix(f"{prefix}_")
    for env in ENV_DIRS:
        if stem.startswith(f"{env}_"):
            return env
    raise ValueError(f"Cannot infer environment from {row_id!r}")


def _summarize(rows_by_split: Mapping[str, List[Dict]], *, name: str, source_dir: Path, env: str) -> Dict:
    return {
        "name": name,
        "source_dir": str(source_dir),
        "environment": env,
        "rows": {split: len(rows) for split, rows in rows_by_split.items()},
        "tool_distribution": {
            split: dict(Counter(_tool(row) for row in rows))
            for split, rows in rows_by_split.items()
        },
    }


def _write_split_dataset(out_dir: Path, rows_by_split: Mapping[str, List[Dict]], summary: Mapping) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        contract = out_dir / f"llm_contract_{split}.jsonl"
        chat = out_dir / f"llm_chat_{split}.jsonl"
        write_jsonl(str(contract), rows)
        validate_dataset_contract_jsonl(str(contract))
        convert_contract_to_qwen_chat_jsonl(str(contract), str(chat))
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def split_source(*, source_dir: Path, prefix: str, out_root: Path) -> Dict[str, Dict[str, int]]:
    rows_by_split = {
        "train": _load_jsonl(source_dir / "llm_contract_train.jsonl"),
        "valid": _load_jsonl(source_dir / "llm_contract_valid.jsonl"),
    }

    results: Dict[str, Dict[str, int]] = {}
    for env, short_env in ENV_DIRS.items():
        env_rows = {
            split: [row for row in rows if _env_from_id(str(row["id"]), prefix) == env]
            for split, rows in rows_by_split.items()
        }
        name = f"{prefix}_{short_env}_consistent_small"
        out_dir = out_root / name
        summary = _summarize(env_rows, name=name, source_dir=source_dir, env=env)
        _write_split_dataset(out_dir, env_rows, summary)
        results[name] = dict(summary["rows"])
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Split the consistent small datasets into environment-specific datasets.")
    ap.add_argument("--woz_source", default="data/woz_consistent_small")
    ap.add_argument("--oracle_source", default="data/oracle_consistent_small")
    ap.add_argument("--out_root", default="data")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    results = {
        "woz": split_source(source_dir=Path(args.woz_source), prefix="woz", out_root=out_root),
        "oracle": split_source(source_dir=Path(args.oracle_source), prefix="oracle", out_root=out_root),
    }
    report_path = out_root / "consistent_small_env_splits_summary.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
