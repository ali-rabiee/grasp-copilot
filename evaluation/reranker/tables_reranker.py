"""Build the headline reranker ablation table from ablation/ + ig_analysis/.

Per plan §7. Reads:
    - evaluation/results/reranker/ablation/<cell>/by_condition.csv
    - evaluation/results/reranker/ablation/<cell>/sweep_meta.json
    - evaluation/results/reranker/ig_analysis/summary.json

Writes:
    - <out_dir>/table_reranker_ablation.{csv,tex}
    - <out_dir>/table_ig_summary.{csv,tex}

Cell name convention: <model_key>__<rerank_mode> (e.g., oracle_woz_lora__info_gain).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


try:
    from evaluation import _bootstrap  # noqa: F401
except Exception:
    import _bootstrap  # type: ignore  # noqa: F401


def _parse_cell_name(name: str) -> Tuple[str, str]:
    if "__" in name:
        m, r = name.rsplit("__", 1)
        return m, r
    return name, "info_gain"


def _per_cell_rollouts(ablation_root: Path) -> Dict[str, Dict[str, Any]]:
    """Return {cell_name: {success_rate, mean_interactions, mean_completion, n_rollouts}}."""
    out: Dict[str, Dict[str, Any]] = {}
    for cell_dir in sorted(p for p in ablation_root.iterdir() if p.is_dir()):
        by_cond = cell_dir / "by_condition.csv"
        if not by_cond.exists():
            continue
        with by_cond.open() as f:
            rows = list(csv.DictReader(f))
        prime_rows = [r for r in rows if r.get("mode") == "prime"]
        if not prime_rows:
            continue
        sr = [float(r["success_rate"]) for r in prime_rows]
        mi = [float(r["mean_interactions"]) for r in prime_rows]
        mc = [float(r["mean_completion_time_sec"]) for r in prime_rows]
        ns = [int(r["n_rollouts"]) for r in prime_rows]
        out[cell_dir.name] = {
            "success_rate": round(st.mean(sr), 4) if sr else 0.0,
            "mean_interactions": round(st.mean(mi), 3) if mi else 0.0,
            "mean_completion_sec": round(st.mean(mc), 3) if mc else 0.0,
            "n_rollouts": sum(ns),
        }
    return out


def _per_cell_ig(summary_path: Path) -> Dict[str, Dict[str, Any]]:
    """Return {cell_name: {mean_ig, frac_ig_ge_0p5}} keyed on selectors.chosen."""
    if not summary_path.exists():
        return {}
    summary = json.loads(summary_path.read_text())
    # The current analyze_ig aggregates across all cells; for per-cell IG, we'd
    # need to re-aggregate from per_question.csv.
    return summary


def _per_cell_ig_from_pq(per_q_csv: Path) -> Dict[str, Dict[str, float]]:
    """Re-aggregate per_question.csv to {cell: {mean_chosen_ig, frac_ge_0p5}}."""
    if not per_q_csv.exists():
        return {}
    by_cell: Dict[str, List[float]] = defaultdict(list)
    with per_q_csv.open() as f:
        for r in csv.DictReader(f):
            if r.get("selector") != "chosen":
                continue
            try:
                by_cell[r["cell"]].append(float(r["ig_bits"]))
            except (KeyError, ValueError):
                continue
    out: Dict[str, Dict[str, float]] = {}
    for cell, vals in by_cell.items():
        if not vals:
            continue
        out[cell] = {
            "mean_ig": round(st.mean(vals), 3),
            "frac_ge_0p5": round(sum(1 for v in vals if v >= 0.5) / len(vals), 3),
        }
    return out


def build_ablation_table(
    ablation_root: Path, ig_analysis_dir: Path, out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = _per_cell_rollouts(ablation_root)
    cell_ig = _per_cell_ig_from_pq(ig_analysis_dir / "per_question.csv")

    fields = ["model", "rerank", "success_pct", "mean_interactions",
              "mean_completion_sec", "mean_ig_bits", "frac_ig_ge_0p5", "n_rollouts"]
    rows: List[Dict[str, Any]] = []
    for cell, agg in sorted(cells.items()):
        model, rerank = _parse_cell_name(cell)
        ig = cell_ig.get(cell, {})
        rows.append({
            "model": model,
            "rerank": rerank,
            "success_pct": f"{100*agg['success_rate']:.1f}",
            "mean_interactions": f"{agg['mean_interactions']:.2f}",
            "mean_completion_sec": f"{agg['mean_completion_sec']:.2f}",
            "mean_ig_bits": f"{ig.get('mean_ig', float('nan')):.3f}" if ig else "—",
            "frac_ig_ge_0p5": f"{ig.get('frac_ge_0p5', float('nan')):.3f}" if ig else "—",
            "n_rollouts": agg["n_rollouts"],
        })

    csv_path = out_dir / "table_reranker_ablation.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # LaTeX (booktabs).
    tex_path = out_dir / "table_reranker_ablation.tex"
    with tex_path.open("w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Information-gain reranker ablation on the noise sweep "
                "(prime mode, all conditions averaged). Best success and mean IG bolded.}\n")
        f.write("\\label{tab:rerank_ablation}\n")
        f.write("\\begin{tabular}{llrrrrr}\n\\toprule\n")
        f.write("Model & Rerank & Success(\\%) & \\#Interact & Time(s) & "
                "Mean IG (b) & Frac IG$\\geq$0.5 \\\\\n\\midrule\n")
        best_success = max((float(r["success_pct"]) for r in rows), default=0.0)
        best_ig = max(
            (float(r["mean_ig_bits"]) for r in rows if r["mean_ig_bits"] != "—"),
            default=0.0,
        )
        for r in rows:
            s = f"{r['success_pct']}"
            if abs(float(s) - best_success) < 1e-6:
                s = f"\\textbf{{{s}}}"
            ig = r["mean_ig_bits"]
            if ig != "—" and abs(float(ig) - best_ig) < 1e-6:
                ig = f"\\textbf{{{ig}}}"
            f.write(f"{r['model']} & {r['rerank']} & {s} & {r['mean_interactions']} "
                    f"& {r['mean_completion_sec']} & {ig} & {r['frac_ig_ge_0p5']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"[tables] wrote {csv_path}")
    print(f"[tables] wrote {tex_path}")


def build_ig_summary_table(ig_analysis_dir: Path, out_dir: Path) -> None:
    summary_path = ig_analysis_dir / "summary.json"
    if not summary_path.exists():
        print(f"[tables] no summary.json at {summary_path}; skipping ig_summary table")
        return
    summary = json.loads(summary_path.read_text())
    selectors = summary.get("selectors", {})

    csv_path = out_dir / "table_ig_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["selector", "n", "mean_ig", "median_ig", "frac_ge_0p5"])
        w.writeheader()
        for sel, s in sorted(selectors.items()):
            w.writerow({
                "selector": sel,
                "n": s.get("n", 0),
                "mean_ig": s.get("mean", 0.0),
                "median_ig": s.get("median", 0.0),
                "frac_ge_0p5": s.get("frac_ge_0p5", 0.0),
            })
    tex_path = out_dir / "table_ig_summary.tex"
    with tex_path.open("w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Per-selector information gain on logged INTERACT calls.}\n")
        f.write("\\label{tab:ig_summary}\n")
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        f.write("Selector & N & Mean IG (b) & Median IG (b) & Frac IG$\\geq$0.5 \\\\\n\\midrule\n")
        for sel, s in sorted(selectors.items()):
            f.write(f"{sel} & {s.get('n', 0)} & {s.get('mean', 0.0):.3f} & "
                    f"{s.get('median', 0.0):.3f} & {s.get('frac_ge_0p5', 0.0):.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"[tables] wrote {csv_path}")
    print(f"[tables] wrote {tex_path}")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ablation_root", default="evaluation/results/reranker/ablation")
    ap.add_argument("--ig_analysis_dir", default="evaluation/results/reranker/ig_analysis")
    ap.add_argument("--out_dir", default="evaluation/results/reranker/tables")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_ablation_table(Path(args.ablation_root), Path(args.ig_analysis_dir), out_dir)
    build_ig_summary_table(Path(args.ig_analysis_dir), out_dir)


if __name__ == "__main__":
    main()
