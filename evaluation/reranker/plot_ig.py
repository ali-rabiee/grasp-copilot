"""Render IG distribution + by-kind facet plots from analyze_ig outputs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


SELECTOR_COLOR = {
    "chosen":    "#2A6FDB",  # blue
    "info_gain": "#2A6FDB",  # alias
    "no_rerank": "#888888",  # grey
    "random":    "#D62728",  # red
    "oracle":    "#2CA02C",  # green
}


def _read_per_question(p: Path) -> List[Dict[str, str]]:
    if not p.exists():
        raise SystemExit(f"per_question.csv not found at {p}")
    with p.open() as f:
        return list(csv.DictReader(f))


def plot_overall(rows: List[Dict[str, str]], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_sel: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        try:
            by_sel[r["selector"]].append(float(r["ig_bits"]))
        except (KeyError, ValueError):
            continue

    if not by_sel:
        print("[plot_ig] no rows to plot")
        return

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bins = [i * 0.25 for i in range(0, 17)]  # 0..4 bits in 0.25-bit bins
    selectors = sorted(by_sel.keys(), key=lambda s: ("chosen" not in s, s))
    for sel in selectors:
        vals = by_sel[sel]
        ax.hist(
            vals, bins=bins, alpha=0.45, density=True,
            label=f"{sel}  (μ={sum(vals)/max(len(vals),1):.2f})",
            color=SELECTOR_COLOR.get(sel, "#444"),
        )
    ax.axvline(0.5, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("Information gain per question (bits)")
    ax.set_ylabel("Density")
    ax.set_title("PRIME IG-reranker — per-question information gain")
    ax.legend(fontsize=8, loc="upper right", frameon=True)
    out_pdf = out_dir / "ig_distribution.pdf"
    out_png = out_dir / "ig_distribution.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_ig] wrote {out_pdf} / {out_png}")


def plot_by_kind(rows: List[Dict[str, str]], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        try:
            by[r.get("interact_kind", "OTHER")][r["selector"]].append(float(r["ig_bits"]))
        except (KeyError, ValueError):
            continue

    kinds = ["QUESTION", "CONFIRM", "SUGGESTION", "OTHER"]
    kinds = [k for k in kinds if k in by]
    if not kinds:
        print("[plot_ig] no kind data")
        return

    fig, axes = plt.subplots(1, len(kinds), figsize=(4.0 * len(kinds), 3.4), sharey=True)
    if len(kinds) == 1:
        axes = [axes]
    bins = [i * 0.25 for i in range(0, 17)]
    for ax, kind in zip(axes, kinds):
        selectors = sorted(by[kind].keys(), key=lambda s: ("chosen" not in s, s))
        for sel in selectors:
            vals = by[kind][sel]
            ax.hist(
                vals, bins=bins, alpha=0.45, density=True,
                label=f"{sel} (μ={sum(vals)/max(len(vals),1):.2f})",
                color=SELECTOR_COLOR.get(sel, "#444"),
            )
        ax.axvline(0.5, color="black", lw=0.8, ls="--", alpha=0.6)
        ax.set_title(kind)
        ax.set_xlabel("IG (bits)")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=7, loc="upper right", frameon=True)
    fig.suptitle("IG per question by INTERACT kind", y=1.02)
    out_pdf = out_dir / "ig_by_kind.pdf"
    out_png = out_dir / "ig_by_kind.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_ig] wrote {out_pdf} / {out_png}")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default="evaluation/results/reranker/ig_analysis")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    rows = _read_per_question(out_dir / "per_question.csv")
    plot_overall(rows, out_dir)
    plot_by_kind(rows, out_dir)


if __name__ == "__main__":
    main()
