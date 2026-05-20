"""
Quick status snapshot of the paper benchmark run.

Usage:
    python -m evaluation.tools.status
    python -m evaluation.tools.status --watch         # refresh every 30 s
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "evaluation" / "results" / "paper_benchmark"

MODEL_ORDER = [
    "oracle_lora",
    "woz_lora",
    "oracle_woz_lora",
    "oracle_woz_r32",
    "oracle_ycb",
    "oracle_stacking",
    "oracle_pouring",
    "qwen3b_zs",
    "h1_ask_if_amb",
    "h2_always_ask",
    "sa1_pred_assist",
    "sa2_bayes_intent",
]
EVAL_ORDER = [
    "oracle_valid_ycb", "oracle_valid_stacking", "oracle_valid_pouring",
    "woz_valid", "ambiguous_ycb", "ambiguous_stacking", "ambiguous_pouring",
]


def _ansi(s: str, code: str) -> str:
    if not os.isatty(1):
        return s
    return f"\033[{code}m{s}\033[0m"


def _human(n: float, unit: str = "s") -> str:
    if unit == "s":
        if n < 60:
            return f"{n:.0f}s"
        if n < 3600:
            return f"{n/60:.1f}m"
        return f"{n/3600:.1f}h"
    return f"{n:.0f}{unit}"


def _matrix(cells: List[Path]) -> Dict[Tuple[str, str], Dict]:
    out: Dict[Tuple[str, str], Dict] = {}
    for p in cells:
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            out[(d.get("_model_safe", ""), d.get("_eval_set", ""))] = d
        except Exception:
            continue
    return out


def _pgrep_runner() -> List[Tuple[int, float, float]]:
    """Return [(pid, etime_sec, cpu_pct), ...] for run_paper_benchmark processes."""
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid,etimes,pcpu,cmd"], text=True
        )
    except Exception:
        return []
    rows = []
    for line in out.splitlines()[1:]:
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        cmd = parts[3]
        if "run_paper_benchmark" in cmd and "python" in cmd:
            try:
                rows.append((int(parts[0]), float(parts[1]), float(parts[2])))
            except Exception:
                continue
    return rows


def _gpu_state() -> str:
    if not shutil.which("nvidia-smi"):
        return "no nvidia-smi"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"],
            text=True,
        ).strip()
        return out
    except Exception:
        return "??"


def print_status() -> None:
    bench = BENCH_DIR
    results = bench / "results"
    cells = sorted(results.glob("*.json")) if results.exists() else []
    matrix = _matrix(cells)

    procs = _pgrep_runner()
    gpu = _gpu_state()

    print(_ansi("=== PRIME paper benchmark — snapshot ===", "1;36"))
    print(f"  bench dir : {bench.relative_to(REPO_ROOT)}")
    print(f"  cells     : {len(cells)} / {len(MODEL_ORDER) * len(EVAL_ORDER)}")
    print(f"  gpu       : {gpu}")
    if procs:
        for pid, etime, cpu in procs:
            print(f"  process   : pid={pid}  elapsed={_human(etime)}  cpu={cpu:.0f}%")
    else:
        print("  process   : (no run_paper_benchmark process found)")

    # Coverage grid: rows = models, cols = eval sets
    print("\n" + _ansi("Coverage grid:", "1"))
    col_w = 4
    header = "  " + f"{'model':<22}" + " ".join(f"{e[:col_w]:>{col_w}}" for e in EVAL_ORDER) + "   acc(avg-3)"
    print(_ansi(header, "2"))
    for m in MODEL_ORDER:
        cells_for_model = sum(1 for e in EVAL_ORDER if (m, e) in matrix)
        if cells_for_model == 0 and m not in {n for (n, _) in matrix}:
            # Hide unloaded models
            continue
        row = f"  {m:<22}"
        accs = []
        for e in EVAL_ORDER:
            cell = matrix.get((m, e))
            if cell is None:
                row += " " + " " * (col_w - 1) + "."
            else:
                acc = cell.get("tool_accuracy", 0) * 100
                row += " " + f"{acc:{col_w}.0f}"
                if e.startswith("oracle_valid_"):
                    accs.append(acc)
        if accs:
            row += f"   {sum(accs)/len(accs):>5.1f}"
        print(row)

    # Latest cell + ETA
    if cells:
        latest = max(cells, key=lambda p: p.stat().st_mtime)
        try:
            with open(latest, "r", encoding="utf-8") as f:
                d = json.load(f)
            t = d.get("timing", {})
            print()
            print(_ansi("Latest cell:", "1"),
                  f"{latest.name}  tool_acc={d.get('tool_accuracy', 0)*100:.1f}%  "
                  f"strict={d.get('strict_exact_rate', 0)*100:.1f}%  "
                  f"n={d.get('n')}  eval_s={_human(t.get('eval_s', 0))}  "
                  f"({t.get('examples_per_sec', 0):.2f} ex/s)")
        except Exception:
            pass

    # ETA based on rolling throughput
    # Only counts cells with timing info (i.e., non-cached cells from this process).
    non_cached = []
    for c in cells:
        try:
            with open(c, "r", encoding="utf-8") as f:
                d = json.load(f)
            es = float((d.get("timing") or {}).get("eval_s", 0))
            if es > 1:
                non_cached.append((c.stat().st_mtime, es, d))
        except Exception:
            continue
    if non_cached and procs:
        last_5 = sorted(non_cached, key=lambda r: r[0])[-5:]
        avg_cell_s = sum(es for _, es, _ in last_5) / len(last_5)
        n_remaining = (len(MODEL_ORDER) - 1) * len(EVAL_ORDER) - len(cells)  # excluding zs
        if n_remaining > 0:
            print(_ansi("ETA estimate:", "1"),
                  f"~{_human(avg_cell_s * n_remaining)}  "
                  f"({n_remaining} cells × avg {_human(avg_cell_s)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--watch", action="store_true", help="Refresh every 30 s")
    args = ap.parse_args()
    if args.watch:
        try:
            while True:
                os.system("clear" if os.name != "nt" else "cls")
                print_status()
                print("\n(watching — Ctrl-C to exit)")
                time.sleep(30)
        except KeyboardInterrupt:
            pass
    else:
        print_status()


if __name__ == "__main__":
    main()
