from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from data_generator.generate_dataset import generate
from llm.data import convert_contract_to_qwen_chat_jsonl, validate_dataset_contract_jsonl, write_jsonl
from llm.prepare_woz_phase2_data import DEFAULT_INSTRUCTION


ENVS = ("reach_to_grasp_ycb", "cube_stacking", "pouring")
TOOLS = ("INTERACT", "APPROACH", "ALIGN_YAW", "STACK", "GRAB", "POUR")
OBJECT_KEYS = ("id", "label", "cell", "yaw", "is_held", "kind", "fill", "stacked_on", "top_of_stack")
MEMORY_KEYS = (
    "n_interactions",
    "past_dialogs",
    "candidates",
    "last_tool_calls",
    "excluded_obj_ids",
    "last_action",
    "last_prompt",
)

TARGETS: Dict[str, Dict[str, Dict[str, int]]] = {
    "train": {
        "reach_to_grasp_ycb": {"INTERACT": 140, "APPROACH": 40, "ALIGN_YAW": 20},
        "cube_stacking": {"INTERACT": 160, "STACK": 40},
        "pouring": {"INTERACT": 140, "GRAB": 40, "POUR": 20},
    },
    "valid": {
        "reach_to_grasp_ycb": {"INTERACT": 24, "APPROACH": 7, "ALIGN_YAW": 3},
        "cube_stacking": {"INTERACT": 26, "STACK": 7},
        "pouring": {"INTERACT": 23, "GRAB": 7, "POUR": 3},
    },
}

ORACLE_TRAIN_SCALE = 5
ORACLE_VALID_SCALE = 3


def _loads(s: str) -> Any:
    return json.loads(s)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _tool(row: Mapping[str, Any]) -> str:
    return str(_loads(str(row["output"]))["tool"])


def _woz_env(row_id: str) -> str:
    for env in ENVS:
        if row_id.startswith(env):
            return env
    raise ValueError(f"Cannot infer WOZ env from id {row_id!r}")


def _normalize_dialogs(past: Iterable[Mapping[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for turn in past or []:
        if "role" in turn and "content" in turn:
            out.append({"role": str(turn["role"]), "content": str(turn["content"])})
            continue
        prompt = turn.get("prompt")
        reply = turn.get("reply")
        if prompt is not None:
            out.append({"role": "assistant", "content": str(prompt)})
        if reply is not None:
            out.append({"role": "user", "content": str(reply)})
    return out[-12:]


def _normalize_input(input_obj: Mapping[str, Any]) -> Dict[str, Any]:
    objects: List[Dict[str, Any]] = []
    for obj in input_obj.get("objects") or []:
        objects.append({key: obj.get(key) for key in OBJECT_KEYS})

    mem_in = dict(input_obj.get("memory") or {})
    memory = {
        "n_interactions": int(mem_in.get("n_interactions") or 0),
        "past_dialogs": _normalize_dialogs(mem_in.get("past_dialogs") or []),
        "candidates": list(mem_in.get("candidates") or []),
        "last_tool_calls": list(mem_in.get("last_tool_calls") or []),
        "excluded_obj_ids": list(mem_in.get("excluded_obj_ids") or []),
        "last_action": dict(mem_in.get("last_action") or {}),
        "last_prompt": dict(mem_in.get("last_prompt") or {}),
    }

    return {
        "objects": objects,
        "gripper_hist": list(input_obj.get("gripper_hist") or []),
        "memory": {key: memory[key] for key in MEMORY_KEYS},
        "user_state": {"mode": "translation"},
    }


def _normalize_output(output_obj: Mapping[str, Any]) -> Dict[str, Any]:
    tool = str(output_obj["tool"])
    args = dict(output_obj.get("args") or {})
    if tool == "INTERACT":
        args = {"kind": args.get("kind"), "text": args.get("text"), "choices": list(args.get("choices") or [])}
    elif tool == "POUR":
        args = {"obj": args.get("obj"), "amount": args.get("amount")}
    else:
        args = {"obj": args.get("obj")}
    return {"tool": tool, "args": args}


def _normalize_contract_row(row: Mapping[str, Any], *, row_id: str) -> Dict[str, str]:
    return {
        "id": row_id,
        "instruction": DEFAULT_INSTRUCTION,
        "input": _dumps(_normalize_input(_loads(str(row["input"])))),
        "output": _dumps(_normalize_output(_loads(str(row["output"])))),
    }


def _generator_to_contract(records: Iterable[Mapping[str, Any]], *, env: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for line_no, rec in enumerate(records, start=1):
        input_obj = {
            "objects": rec["objects"],
            "gripper_hist": rec["gripper_hist"],
            "memory": rec["memory"],
            "user_state": rec["user_state"],
        }
        row = {
            "input": _dumps(input_obj),
            "output": _dumps(rec["target_tool_call"]),
        }
        rows.append(_normalize_contract_row(row, row_id=f"oracle_{env}_{rec['episode_id']}_{line_no}"))
    return rows


def _load_woz_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(_normalize_contract_row(row, row_id=f"woz_{row['id']}"))
    return rows


def _row_env(row: Mapping[str, str]) -> str:
    row_id = row["id"]
    if row_id.startswith("woz_"):
        return _woz_env(row_id.removeprefix("woz_"))
    for env in ENVS:
        if row_id.startswith(f"oracle_{env}_"):
            return env
    raise ValueError(f"Cannot infer env from id {row_id!r}")


def _bucket(rows: Iterable[Dict[str, str]]) -> Dict[Tuple[str, str], List[Dict[str, str]]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        inp = _loads(row["input"])
        if inp.get("user_state", {}).get("mode") != "translation":
            continue
        tool = _tool(row)
        if tool not in TOOLS:
            continue
        buckets[(_row_env(row), tool)].append(row)
    return buckets


def _scaled_targets(*, train_scale: int, valid_scale: int) -> Dict[str, Dict[str, Dict[str, int]]]:
    return {
        split: {
            env: {
                tool: count * (train_scale if split == "train" else valid_scale)
                for tool, count in counts.items()
            }
            for env, counts in envs.items()
        }
        for split, envs in TARGETS.items()
    }


def _sample_split(
    rows: Iterable[Dict[str, str]],
    *,
    targets: Mapping[str, Mapping[str, Mapping[str, int]]],
    seed: int,
) -> Dict[str, List[Dict[str, str]]]:
    rng = random.Random(seed)
    buckets = _bucket(rows)
    out = {"train": [], "valid": []}

    for env in ENVS:
        for tool in TOOLS:
            needed = int(targets["train"].get(env, {}).get(tool, 0)) + int(targets["valid"].get(env, {}).get(tool, 0))
            if needed == 0:
                continue
            key = (env, tool)
            choices = list(buckets.get(key) or [])
            if len(choices) < needed:
                raise ValueError(f"Need {needed} rows for {env}/{tool}; found {len(choices)}")
            rng.shuffle(choices)
            n_train = int(targets["train"].get(env, {}).get(tool, 0))
            n_valid = int(targets["valid"].get(env, {}).get(tool, 0))
            out["train"].extend(choices[:n_train])
            out["valid"].extend(choices[n_train : n_train + n_valid])

    rng.shuffle(out["train"])
    rng.shuffle(out["valid"])
    return out


def _write_dataset(out_dir: Path, splits: Mapping[str, List[Dict[str, str]]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        contract = out_dir / f"llm_contract_{split}.jsonl"
        chat = out_dir / f"llm_chat_{split}.jsonl"
        write_jsonl(str(contract), rows)
        validate_dataset_contract_jsonl(str(contract))
        convert_contract_to_qwen_chat_jsonl(str(contract), str(chat))


def _dist(rows: Iterable[Mapping[str, str]], key: str) -> Counter[str]:
    c: Counter[str] = Counter()
    for row in rows:
        inp = _loads(row["input"])
        out = _loads(row["output"])
        if key == "env":
            c[_row_env(row)] += 1
        elif key == "tool":
            c[str(out["tool"])] += 1
        elif key == "mode":
            c[str(inp["user_state"]["mode"])] += 1
        elif key == "interact_kind" and out["tool"] == "INTERACT":
            c[str(out["args"]["kind"])] += 1
        elif key == "object_shape":
            for obj in inp.get("objects") or []:
                c[",".join(sorted(obj.keys()))] += 1
        elif key == "dialog_shape":
            for dialog in inp.get("memory", {}).get("past_dialogs") or []:
                c[",".join(sorted(dialog.keys()))] += 1
    return c


def _props(counter: Counter[str]) -> Dict[str, float]:
    total = sum(counter.values()) or 1
    return {k: round(v / total, 6) for k, v in sorted(counter.items())}


def _max_prop_delta(a: Counter[str], b: Counter[str]) -> float:
    ap, bp = _props(a), _props(b)
    keys = set(ap) | set(bp)
    return round(max(abs(ap.get(k, 0.0) - bp.get(k, 0.0)) for k in keys) if keys else 0.0, 6)


def _summary(name: str, splits: Mapping[str, List[Dict[str, str]]]) -> Dict[str, Any]:
    rows = [row for split_rows in splits.values() for row in split_rows]
    return {
        "name": name,
        "rows": {split: len(split_rows) for split, split_rows in splits.items()},
        "tool_distribution": {split: dict(_dist(split_rows, "tool")) for split, split_rows in splits.items()},
        "env_distribution": {split: dict(_dist(split_rows, "env")) for split, split_rows in splits.items()},
        "mode_distribution": {split: dict(_dist(split_rows, "mode")) for split, split_rows in splits.items()},
        "all_tools": dict(_dist(rows, "tool")),
        "object_shapes": dict(_dist(rows, "object_shape")),
        "dialog_shapes": dict(_dist(rows, "dialog_shape")),
    }


def _similarity_report(
    woz: Mapping[str, List[Dict[str, str]]],
    oracle: Mapping[str, List[Dict[str, str]]],
    *,
    oracle_episodes_per_env: int,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "oracle_episodes_per_env": oracle_episodes_per_env,
        "format_checks": {},
        "distribution_similarity": {},
    }
    for split in ("train", "valid"):
        w_rows, o_rows = woz[split], oracle[split]
        report["distribution_similarity"][split] = {
            key: {
                "woz": _props(_dist(w_rows, key)),
                "oracle": _props(_dist(o_rows, key)),
                "max_abs_prop_delta": _max_prop_delta(_dist(w_rows, key), _dist(o_rows, key)),
            }
            for key in ("env", "tool", "mode", "interact_kind", "object_shape", "dialog_shape")
        }

    for name, splits in (("woz", woz), ("oracle", oracle)):
        all_rows = [row for split_rows in splits.values() for row in split_rows]
        report["format_checks"][name] = {
            "top_level_keys": sorted({tuple(sorted(row.keys())) for row in all_rows}),
            "instruction_count": len({row["instruction"] for row in all_rows}),
            "input_keys": sorted({tuple(sorted(_loads(row["input"]).keys())) for row in all_rows}),
            "memory_keys": sorted({tuple(sorted(_loads(row["input"])["memory"].keys())) for row in all_rows}),
            "output_tools": sorted(_dist(all_rows, "tool")),
            "choice_limit_violations": sum(
                1
                for row in all_rows
                for out in [_loads(row["output"])]
                if out["tool"] == "INTERACT" and len(out["args"].get("choices") or []) > 5
            ),
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Build matched small WOZ and oracle datasets.")
    ap.add_argument("--woz_source", default="data/woz_phase2/llm_contract_all.jsonl")
    ap.add_argument("--woz_out", default="data/woz_consistent_small")
    ap.add_argument("--oracle_out", default="data/oracle_consistent_small")
    ap.add_argument("--report_out", default="data/consistent_small_similarity_report.json")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--oracle_episodes_per_env", type=int, default=2500)
    args = ap.parse_args()

    woz_rows = _load_woz_rows(Path(args.woz_source))
    oracle_rows: List[Dict[str, str]] = []
    oracle_stats: Dict[str, Any] = {}
    for env in ENVS:
        records, stats = generate(episodes=int(args.oracle_episodes_per_env), seed=int(args.seed), env=env)
        oracle_rows.extend(_generator_to_contract(records, env=env))
        oracle_stats[env] = stats

    woz_splits = _sample_split(woz_rows, targets=TARGETS, seed=int(args.seed))
    oracle_splits = _sample_split(
        oracle_rows,
        targets=_scaled_targets(train_scale=ORACLE_TRAIN_SCALE, valid_scale=ORACLE_VALID_SCALE),
        seed=int(args.seed),
    )

    _write_dataset(Path(args.woz_out), woz_splits)
    _write_dataset(Path(args.oracle_out), oracle_splits)

    woz_summary = _summary("woz_consistent_small", woz_splits)
    woz_summary["source"] = str(args.woz_source)
    oracle_summary = _summary("oracle_consistent_small", oracle_splits)
    oracle_summary["source"] = {
        "generator": "data_generator.generate_dataset.generate",
        "episodes_per_env": int(args.oracle_episodes_per_env),
        "seed": int(args.seed),
        "stats_by_env": oracle_stats,
    }

    with (Path(args.woz_out) / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(woz_summary, f, indent=2, sort_keys=True)
    with (Path(args.oracle_out) / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(oracle_summary, f, indent=2, sort_keys=True)

    report = _similarity_report(
        woz_splits,
        oracle_splits,
        oracle_episodes_per_env=int(args.oracle_episodes_per_env),
    )
    with Path(args.report_out).open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(json.dumps({"woz": woz_summary, "oracle": oracle_summary, "similarity": report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
