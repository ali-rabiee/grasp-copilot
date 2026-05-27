#!/usr/bin/env python3
"""Print the Qwen2.5-7B (ZS) row for Table 1 of the CoRL paper.

Reads per-cell JSON outputs at
  evaluation/results/paper_benchmark/results/qwen7b_zs__<eval>.json
and prints a LaTeX row matching the existing Table 1 format:

  Qwen2.5-7B (ZS) & YCB & Stk & Pour & Avg-3 & WoZ \\

Usage:
  python3 scripts/print_qwen7b_zs_table_row.py \
      [--results_dir evaluation/results/paper_benchmark/results]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EVAL_SETS = ["oracle_valid_ycb", "oracle_valid_stacking", "oracle_valid_pouring", "woz_valid"]
LABELS = {"oracle_valid_ycb": "YCB", "oracle_valid_stacking": "Stk", "oracle_valid_pouring": "Pour", "woz_valid": "WoZ"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", default="evaluation/results/paper_benchmark/results")
    ap.add_argument("--model_safe", default="qwen7b_zs")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    cells = {}
    for ev in EVAL_SETS:
        p = rd / f"{args.model_safe}__{ev}.json"
        if not p.exists():
            print(f"MISSING {p}")
            continue
        with open(p) as f:
            d = json.load(f)
        cells[ev] = {
            "n": d["n"],
            "tool_acc": d["tool_accuracy"],
            "json_valid": d["json_valid_rate"],
            "schema_valid": d["schema_valid_rate"],
        }
        print(f"{ev}: n={d['n']}  tool_acc={d['tool_accuracy']*100:.2f}%  json_valid={d['json_valid_rate']*100:.1f}%  schema_valid={d['schema_valid_rate']*100:.1f}%")

    clean = [cells[k]["tool_acc"] for k in ("oracle_valid_ycb", "oracle_valid_stacking", "oracle_valid_pouring") if k in cells]
    if len(clean) == 3:
        avg3 = sum(clean) / 3.0
        ycb = cells["oracle_valid_ycb"]["tool_acc"] * 100
        stk = cells["oracle_valid_stacking"]["tool_acc"] * 100
        pour = cells["oracle_valid_pouring"]["tool_acc"] * 100
        woz = cells["woz_valid"]["tool_acc"] * 100 if "woz_valid" in cells else None
        print()
        print("Table 1 row (insert after 'Qwen2.5-3B (ZS)'):")
        if woz is not None:
            print(f"Qwen2.5-7B (ZS) & {ycb:.1f} & {stk:.1f} & {pour:.1f} & {avg3*100:.1f} & {woz:.1f} \\\\")
        else:
            print(f"Qwen2.5-7B (ZS) & {ycb:.1f} & {stk:.1f} & {pour:.1f} & {avg3*100:.1f} & -- \\\\")


if __name__ == "__main__":
    main()
