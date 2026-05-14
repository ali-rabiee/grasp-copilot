from __future__ import annotations

import argparse
import json
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .utils import json_loads_strict


ACTION_TOOLS = {"APPROACH", "ALIGN_YAW", "STACK", "GRAB", "POUR", "RELEASE"}


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _tool_from_output_str(output_str: str) -> Optional[str]:
    try:
        obj = json_loads_strict(output_str)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    t = obj.get("tool")
    return str(t) if isinstance(t, str) else None


def rebalance_contract(
    *,
    in_path: str,
    out_path: str,
    seed: int = 0,
    motion_repeat: int = 10,
    interact_keep_prob: float = 1.0,
) -> Dict[str, int]:
    """
    Create a rebalanced contract JSONL from an existing contract JSONL.

    Why this exists:
      The dataset is naturally INTERACT-heavy. That matches reality, but models then
      often learn a "safe default" of emitting INTERACT even when an action tool is
      correct. Repeating (or upweighting) motion rows is a simple, effective fix.

    This keeps the *same* examples, just changes sampling/duplication.

    Args:
      motion_repeat: repeat each action-tool example this many times.
      interact_keep_prob: probability of keeping each INTERACT example (1.0 keeps all).
    """
    rng = random.Random(int(seed))
    stats = {"kept_interact": 0, "kept_motion": 0, "dropped_interact": 0, "unknown": 0, "written": 0}

    out_rows: List[Dict[str, Any]] = []
    for r in _iter_jsonl(in_path):
        tool = _tool_from_output_str(str(r.get("output", "")))
        if tool in ACTION_TOOLS:
            stats["kept_motion"] += 1
            rep = max(1, int(motion_repeat))
            for k in range(rep):
                rr = dict(r)
                # Make ids unique for datasets/trl bookkeeping.
                rr["id"] = f"{r.get('id','')}_m{k}"
                out_rows.append(rr)
        elif tool == "INTERACT":
            if float(interact_keep_prob) >= 1.0 or rng.random() < float(interact_keep_prob):
                stats["kept_interact"] += 1
                out_rows.append(r)
            else:
                stats["dropped_interact"] += 1
        else:
            stats["unknown"] += 1

    rng.shuffle(out_rows)
    stats["written"] = len(out_rows)
    _write_jsonl(out_path, out_rows)
    return stats


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Rebalance a contract JSONL to emphasize action-tool examples.")
    ap.add_argument("--in_contract", type=str, required=True)
    ap.add_argument("--out_contract", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--motion_repeat", type=int, default=10, help="Repeat action-tool examples N times.")
    ap.add_argument(
        "--interact_keep_prob",
        type=float,
        default=1.0,
        help="Keep probability for INTERACT examples (1.0 keeps all, <1.0 downsamples).",
    )
    args = ap.parse_args(argv)

    stats = rebalance_contract(
        in_path=str(args.in_contract),
        out_path=str(args.out_contract),
        seed=int(args.seed),
        motion_repeat=int(args.motion_repeat),
        interact_keep_prob=float(args.interact_keep_prob),
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

