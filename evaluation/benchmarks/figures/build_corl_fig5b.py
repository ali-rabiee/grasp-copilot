"""Build the user-input-noise figure for the CoRL paper (Figure 5b).

Two panels:
  (a) Task success rate by noise condition for PRIME (Oracle+WoZ v2) vs.
      rule-based shared-autonomy baselines in PRIME mode.
      Headline: heuristics collapse to 0%; PRIME stays high.
  (b) PRIME interactions per trial by noise condition.
      Headline: question rate adapts to noise (~5 clean -> ~12-20 under noise).

Usage:
    python -m evaluation.benchmarks.figures.build_corl_fig5b \\
        --sweeps_root evaluation/results/robustness/user_input_noise/sweeps \\
        --out_dir A_Rabiee_corl_2026_PRIME/figs \\
        --out_basename fig5b_user_input_noise
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

CONDITION_ORDER = ["clean", "dir_low", "dir_high", "sel_low", "sel_high", "compound_mid"]
CONDITION_LABELS = {
    "clean": "clean",
    "dir_low": "dir-low",
    "dir_high": "dir-high",
    "sel_low": "sel-low",
    "sel_high": "sel-high",
    "compound_mid": "compound",
}

# Okabe-Ito palette
COLOR_PRIME = "#0072B2"  # blue
COLOR_HEURISTIC = "#D55E00"  # vermillion

SYSTEMS = {
    "oracle_woz_v2": ("Oracle+WoZ", COLOR_PRIME),
    "h1_ask_if_amb":   ("H1",  COLOR_HEURISTIC),
    "sa1_pred_assist": ("SA1", "#E69F00"),
    "sa2_bayes_intent":("SA2", "#56B4E9"),
}


def load_by_condition(sweeps_root: Path, sys_key: str) -> List[dict]:
    """Locate the by_condition.csv for a given system, handling nested directories."""
    candidates = [
        sweeps_root / sys_key / "by_condition.csv",
        sweeps_root / sys_key / sys_key / "by_condition.csv",
    ]
    for c in candidates:
        if c.exists():
            with c.open() as fh:
                return list(csv.DictReader(fh))
    raise FileNotFoundError(f"No by_condition.csv for {sys_key} under {sweeps_root}")


def aggregate_prime_mode(rows: List[dict], difficulty: str = None) -> Dict[str, Dict[str, float]]:
    """Aggregate prime-mode rows by condition. If difficulty is set ('easy' / 'hard'),
    restrict to that subset; otherwise weight easy + hard by n_rollouts."""
    out: Dict[str, Dict[str, float]] = {}
    for cond in CONDITION_ORDER:
        rs = [r for r in rows if r["mode"] == "prime" and r["condition"] == cond
              and (difficulty is None or r["difficulty"] == difficulty)]
        if not rs:
            continue
        n_total = sum(int(r["n_rollouts"]) for r in rs)
        if n_total == 0:
            continue
        succ = sum(float(r["success_rate"]) * int(r["n_rollouts"]) for r in rs) / n_total
        inter = sum(float(r["mean_interactions"]) * int(r["n_rollouts"]) for r in rs) / n_total
        out[cond] = {"success_rate": succ, "mean_interactions": inter, "n": n_total}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweeps_root", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--out_basename", default="fig5b_user_input_noise")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Collect aggregates for every system that has data.
    # Panel (a) — success rate — uses easy+hard aggregate so the heuristic-collapse story
    # spans both difficulties. Panel (b) — interaction rate — uses easy-only because
    # hard scenarios already cap out the interaction budget on clean (baseline task is
    # hard enough to saturate PRIME's question count regardless of noise).
    agg_all: Dict[str, Dict[str, Dict[str, float]]] = {}
    agg_easy: Dict[str, Dict[str, Dict[str, float]]] = {}
    for sys_key in SYSTEMS:
        try:
            rows = load_by_condition(args.sweeps_root, sys_key)
            agg_all[sys_key]  = aggregate_prime_mode(rows, difficulty=None)
            agg_easy[sys_key] = aggregate_prime_mode(rows, difficulty="easy")
            print(f"loaded {sys_key}: all={len(agg_all[sys_key])} cond / easy={len(agg_easy[sys_key])}")
        except FileNotFoundError as e:
            print(f"  warn: {e}")
    aggregates = agg_all  # for panel (a)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))

    x = np.arange(len(CONDITION_ORDER))
    width = 0.14

    # Panel (a): success rate
    for i, (sys_key, (label, color)) in enumerate(SYSTEMS.items()):
        if sys_key not in aggregates:
            continue
        vals = [aggregates[sys_key].get(c, {}).get("success_rate", np.nan) * 100
                for c in CONDITION_ORDER]
        offset = (i - (len(SYSTEMS) - 1) / 2) * width
        ax1.bar(x + offset, vals, width=width, color=color, label=label, edgecolor="white", linewidth=0.4)

    ax1.set_xticks(x)
    ax1.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER], rotation=20)
    ax1.set_ylabel("Task success rate (\\%)")
    ax1.set_title("(a) Heuristic baselines collapse in PRIME mode")
    ax1.set_ylim(0, 105)
    ax1.legend(loc="upper right", fontsize=8, ncol=2, frameon=False)
    ax1.grid(axis="y", linestyle=":", alpha=0.4)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel (b): PRIME interaction count by condition (easy trials only)
    if "oracle_woz_v2" in agg_easy:
        prime_inter = [agg_easy["oracle_woz_v2"].get(c, {}).get("mean_interactions", np.nan)
                       for c in CONDITION_ORDER]
        ax2.bar(x, prime_inter, color=COLOR_PRIME, edgecolor="white", linewidth=0.4)
        ax2.axhline(prime_inter[0] if not np.isnan(prime_inter[0]) else 0,
                    color="grey", linestyle="--", linewidth=0.8, alpha=0.7,
                    label=f"clean baseline = {prime_inter[0]:.1f}")
        ax2.legend(loc="upper left", fontsize=8, frameon=False)

    ax2.set_xticks(x)
    ax2.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER], rotation=20)
    ax2.set_ylabel("Mean PRIME interactions per trial")
    ax2.set_title("(b) PRIME asks more questions as noise rises (easy trials)")
    ax2.grid(axis="y", linestyle=":", alpha=0.4)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()

    pdf_path = args.out_dir / f"{args.out_basename}.pdf"
    png_path = args.out_dir / f"{args.out_basename}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPDF: {pdf_path}")
    print(f"PNG: {png_path}")

    # Headline numbers
    if "oracle_woz_v2" in aggregates:
        print("\n=== PRIME (Oracle+WoZ v2) - all difficulties (panel a) ===")
        for c in CONDITION_ORDER:
            d = aggregates["oracle_woz_v2"].get(c)
            if d:
                print(f"  {c:14s}: success={d['success_rate']*100:5.1f}%  inter={d['mean_interactions']:6.2f}")
        print("\n=== PRIME — easy only (panel b adaptation story) ===")
        for c in CONDITION_ORDER:
            d = agg_easy["oracle_woz_v2"].get(c)
            if d:
                print(f"  {c:14s}: success={d['success_rate']*100:5.1f}%  inter={d['mean_interactions']:6.2f}")


if __name__ == "__main__":
    main()
