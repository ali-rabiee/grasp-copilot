"""Build the IG-bits histogram for the CoRL paper (Figure 7, panel c).

Reads `dialogs.jsonl` from a reranker ablation cell and produces a histogram of
per-question information gain in bits, faceted by interact_kind.

Headline annotations:
  - Mean IG (overall) with a vertical line.
  - Fraction of questions delivering >= 0.5 bits.
  - Per-kind sub-histograms.

Usage:
    python -m evaluation.reranker.build_ig_histogram \\
        --dialogs evaluation/results/reranker/ablation/oracle_woz_lora__info_gain/dialogs.jsonl \\
        --out_dir A_Rabiee_corl_2026_PRIME/figs \\
        --out_basename fig7c_ig_histogram

Outputs PDF + PNG.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

KIND_ORDER = ["QUESTION", "CONFIRM", "SUGGESTION"]
# Okabe-Ito colorblind-safe palette
KIND_COLORS = {
    "QUESTION":   "#0072B2",  # blue
    "CONFIRM":    "#D55E00",  # vermillion
    "SUGGESTION": "#009E73",  # bluish green
}
GATE_BITS = 0.5


def load_dialogs(path: Path) -> List[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def plot(rows: List[dict], out_path_pdf: Path, out_path_png: Path,
         multi_candidate_only: bool = False) -> dict:
    if multi_candidate_only:
        rows = [r for r in rows if r.get("n_candidates_before", 1) > 1]

    overall_ig = np.array([r.get("ig_bits", 0.0) for r in rows])
    mean_overall = float(np.mean(overall_ig))
    frac_ge_gate = float(np.mean(overall_ig >= GATE_BITS))

    per_kind = {k: [] for k in KIND_ORDER}
    for r in rows:
        k = r.get("interact_kind")
        if k in per_kind:
            per_kind[k].append(r.get("ig_bits", 0.0))

    fig, ax = plt.subplots(figsize=(5.5, 3.4))

    bins = np.linspace(0.0, 2.0, 41)  # 0.05-bit bins, covers up to 2 bits (4-way max)
    bottom = np.zeros(len(bins) - 1)
    for kind in KIND_ORDER:
        vals = np.clip(np.asarray(per_kind[kind]), 0.0, 2.0)
        if len(vals) == 0:
            continue
        counts, _ = np.histogram(vals, bins=bins)
        ax.bar(
            bins[:-1], counts, width=np.diff(bins), bottom=bottom,
            align="edge", color=KIND_COLORS[kind], edgecolor="white",
            linewidth=0.3, label=f"{kind} (n={len(vals)})",
        )
        bottom = bottom + counts

    # Gate marker
    ax.axvline(GATE_BITS, color="black", linestyle="--", linewidth=1.0,
               label=f"gate = {GATE_BITS} bits")
    # Overall mean marker
    ax.axvline(mean_overall, color="#CC79A7", linestyle="-", linewidth=1.5,
               label=f"mean = {mean_overall:.2f} bits")

    ax.set_xlabel("Information gain per emitted question (bits)")
    ax.set_ylabel("Count")
    title = "PRIME question informativeness"
    if multi_candidate_only:
        title += "  (multi-candidate scenes only)"
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_xlim(0.0, 2.05)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Inline annotation
    ax.text(
        0.98, 0.55,
        f"frac $\\geq$ {GATE_BITS} bits: {frac_ge_gate*100:.1f}\\%",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(out_path_pdf, bbox_inches="tight")
    fig.savefig(out_path_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_rows": len(rows),
        "mean_ig_bits": mean_overall,
        "median_ig_bits": float(np.median(overall_ig)),
        "frac_ge_0_5_bits": frac_ge_gate,
        "frac_gt_0_bits": float(np.mean(overall_ig > 0)),
        "per_kind": {
            k: {
                "n": len(v),
                "mean_ig_bits": float(np.mean(v)) if v else None,
                "frac_ge_0_5_bits": float(np.mean(np.array(v) >= GATE_BITS)) if v else None,
            }
            for k, v in per_kind.items()
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dialogs", required=True, type=Path,
                    help="Path to a dialogs.jsonl from a reranker ablation cell.")
    ap.add_argument("--out_dir", required=True, type=Path,
                    help="Directory where the PDF/PNG land.")
    ap.add_argument("--out_basename", default="fig7c_ig_histogram",
                    help="Filename stem (no extension).")
    ap.add_argument("--multi_only", action="store_true",
                    help="Restrict to scenes with >1 candidate before the question.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_dialogs(args.dialogs)
    print(f"loaded {len(rows)} INTERACT logs from {args.dialogs}")

    pdf_path = args.out_dir / f"{args.out_basename}.pdf"
    png_path = args.out_dir / f"{args.out_basename}.png"
    stats = plot(rows, pdf_path, png_path, multi_candidate_only=args.multi_only)

    print(json.dumps(stats, indent=2))
    print(f"\n  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")


if __name__ == "__main__":
    main()
