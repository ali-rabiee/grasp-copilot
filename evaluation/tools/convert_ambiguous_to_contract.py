"""
Convert ambiguous_eval/*/grasp_gen.jsonl into contract JSONL.

Each input record carries fields:
  id, episode_id, env_name, ambiguity_category, wizard_prompt_type,
  ambiguity_reason, objects, gripper_hist, memory, user_state,
  target_tool_call, oracle_baseline_call

We emit one contract row per record with the same {id, instruction, input,
output} layout the offline benchmark consumes. The original id and the
ambiguity_category are preserved so downstream tooling can group by category.

Usage:
    python -m evaluation.tools.convert_ambiguous_to_contract \
        --in_dirs data/ambiguous_eval_reach_to_grasp_ycb \
                  data/ambiguous_eval_cube_stacking \
                  data/ambiguous_eval_pouring \
        --out_dir data/_contracts_ambiguous

By default the script discovers all data/ambiguous_eval_* directories under
the project root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

try:
    from evaluation import _bootstrap  # noqa: F401
except Exception:
    import _bootstrap  # type: ignore  # noqa: F401

from llm.data import iter_jsonl, validate_dataset_contract_jsonl, write_jsonl


INSTRUCTION = (
    "Given the robot observation and dialog context, infer the user's intent and "
    "emit exactly one tool call. Output ONLY the tool call JSON with keys tool and args. "
    "If the tool is INTERACT, you must output at most 5 choices total."
)

REQUIRED_KEYS = (
    "objects", "gripper_hist", "memory", "user_state", "target_tool_call",
)


def convert_one(src: Path, dst: Path, *, max_past_dialogs: int = 12) -> int:
    rows: List[dict] = []
    for line_no, obj in iter_jsonl(str(src)):
        for k in REQUIRED_KEYS:
            if k not in obj:
                raise ValueError(f"{src}:{line_no} missing key {k!r}")

        mem = obj["memory"]
        if isinstance(mem, dict) and isinstance(mem.get("past_dialogs"), list) and int(max_past_dialogs) > 0:
            mem = dict(mem)
            mem["past_dialogs"] = list(mem.get("past_dialogs") or [])[-int(max_past_dialogs):]

        input_blob = {
            "objects": obj["objects"],
            "gripper_hist": obj["gripper_hist"],
            "memory": mem,
            "user_state": obj["user_state"],
        }
        target = obj["target_tool_call"]
        if not isinstance(target, dict):
            raise ValueError(f"{src}:{line_no} target_tool_call must be an object")

        ex_id = obj.get("id") or f"{obj.get('episode_id', 'ep')}_{line_no}"
        rows.append({
            "id": str(ex_id),
            "instruction": INSTRUCTION,
            "input": json.dumps(input_blob, ensure_ascii=False, separators=(",", ":")),
            "output": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
            # Side-channel metadata for grouping at analysis time. These keys
            # are ignored by the benchmark but useful for category breakdowns.
            "meta": {
                "env_name": obj.get("env_name"),
                "ambiguity_category": obj.get("ambiguity_category"),
                "wizard_prompt_type": obj.get("wizard_prompt_type"),
                "oracle_baseline_call": obj.get("oracle_baseline_call"),
            },
        })

    dst.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(dst), rows)
    # Validate without the meta field (validator only checks the contract keys).
    validate_dataset_contract_jsonl(str(dst))
    return len(rows)


def find_ambiguous_dirs(root: Path) -> List[Path]:
    return sorted(p for p in (root / "data").glob("ambiguous_eval_*") if p.is_dir())


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_dirs", nargs="*", default=None,
                    help="Source directories containing grasp_gen.jsonl. Defaults to discovering data/ambiguous_eval_*.")
    ap.add_argument("--out_dir", default="data/_contracts_ambiguous",
                    help="Where to write the contract JSONLs (one per source dir).")
    ap.add_argument("--root", default=None, help="Project root override.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.root:
        root = Path(args.root).resolve()
    else:
        here = Path(__file__).resolve().parent
        root = here.parent

    if args.in_dirs:
        sources = [Path(d) for d in args.in_dirs]
    else:
        sources = find_ambiguous_dirs(root)
    if not sources:
        raise SystemExit("No ambiguous_eval_* directories found.")

    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = root / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    written = []
    for src_dir in sources:
        src = src_dir / "grasp_gen.jsonl"
        if not src.exists():
            print(f"[skip] {src} missing")
            continue
        # Normalize: data/ambiguous_eval_reach_to_grasp_ycb -> ambiguous_reach_to_grasp_ycb.jsonl
        stem = src_dir.name.replace("ambiguous_eval_", "")
        dst = out_root / f"ambiguous_{stem}.jsonl"
        n = convert_one(src, dst)
        written.append((str(dst), n))
        print(f"[ok] {src} -> {dst} ({n} rows)")

    manifest = out_root / "manifest.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({"sources": [str(s) for s in sources], "outputs": written}, f, indent=2)
    print(f"[manifest] {manifest}")


if __name__ == "__main__":
    main()
