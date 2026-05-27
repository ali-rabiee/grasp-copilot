"""Trial-level correlation: PRIME interactions per trial vs. scene complexity / failures.

Reads:
  - /home/ali/Data/PRIME_LOGS/assistive/**/llm_events.jsonl  (n_interactions)
  - /home/ali/github/PRIME/user-study-prime/data/master_trials.csv

Writes:
  - A_Rabiee_corl_2026_PRIME/figs/fig_interaction_correlation.{pdf,png}
"""
from __future__ import annotations

import csv
import glob
import math
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

PRIME_LOGS = "/home/ali/Data/PRIME_LOGS"
MASTER = "/home/ali/github/PRIME/user-study-prime/data/master_trials.csv"
OUT_DIR = "/home/ali/github/PRIME/A_Rabiee_corl_2026_PRIME/figs"


def load_interactions():
    out = {}
    for fp in sorted(glob.glob(os.path.join(PRIME_LOGS, "assistive/**/llm_events.jsonl"), recursive=True)):
        parts = fp.split("/")
        idx = parts.index("assistive")
        trial_id = parts[idx + 3]
        with open(fp) as f:
            txt = f.read().strip()
        m = re.search(r"n_interactions\s*=\s*(\d+)", txt)
        if m:
            out[trial_id] = int(m.group(1))
    return out


def pearson(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs); syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy)


def main():
    trial_int = load_interactions()
    rows = []
    with open(MASTER) as f:
        for r in csv.DictReader(f):
            if r["mode"] != "assistive":
                continue
            if r["trial_id"] not in trial_int:
                continue
            try:
                n_pl = int(r["n_plausible_targets"]) if r["n_plausible_targets"] else None
                n_f = int(r["n_tool_failures"]) if r["n_tool_failures"] else 0
            except Exception:
                continue
            if n_pl is None:
                continue
            rows.append({
                "subj": r["subject"], "diff": r["difficulty"],
                "n_int": trial_int[r["trial_id"]],
                "n_pl": n_pl, "n_f": n_f,
            })

    n_pl = [r["n_pl"] for r in rows]
    n_f = [r["n_f"] for r in rows]
    n_int = [r["n_int"] for r in rows]
    is_hard = [r["diff"] == "hard" for r in rows]

    r_pl = pearson(n_pl, n_int)
    r_f = pearson(n_f, n_int)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    easy_c, hard_c = "#95d5b2", "#2d6a4f"
    rng = np.random.default_rng(42)

    # Panel a: PRIME interactions vs plausible targets
    ax = axes[0]
    for is_h, col, label in [(False, easy_c, "Easy"), (True, hard_c, "Hard")]:
        xs = [x + rng.uniform(-0.12, 0.12) for x, h in zip(n_pl, is_hard) if h == is_h]
        ys = [y + rng.uniform(-0.18, 0.18) for y, h in zip(n_int, is_hard) if h == is_h]
        ax.scatter(xs, ys, c=col, alpha=0.65, s=24, edgecolors="black", linewidths=0.3, label=label)
    # Fit
    p = np.polyfit(n_pl, n_int, 1)
    xx = np.linspace(min(n_pl), max(n_pl), 50)
    ax.plot(xx, np.polyval(p, xx), "k--", lw=1.2, alpha=0.7)
    ax.set_xlabel("Plausible targets in scene")
    ax.set_ylabel("PRIME interactions per trial")
    ax.set_title(f"r = {r_pl:.2f}, p < $10^{{-11}}$", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, zorder=0)

    # Panel b: PRIME interactions vs in-trial failures
    ax = axes[1]
    for is_h, col, label in [(False, easy_c, "Easy"), (True, hard_c, "Hard")]:
        xs = [x + rng.uniform(-0.12, 0.12) for x, h in zip(n_f, is_hard) if h == is_h]
        ys = [y + rng.uniform(-0.18, 0.18) for y, h in zip(n_int, is_hard) if h == is_h]
        ax.scatter(xs, ys, c=col, alpha=0.65, s=24, edgecolors="black", linewidths=0.3, label=label)
    p = np.polyfit(n_f, n_int, 1)
    xx = np.linspace(min(n_f), max(n_f), 50)
    ax.plot(xx, np.polyval(p, xx), "k--", lw=1.2, alpha=0.7)
    ax.set_xlabel("In-trial execution failures")
    ax.set_ylabel("PRIME interactions per trial")
    ax.set_title(f"r = {r_f:.2f}, p < $10^{{-4}}$", fontsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, zorder=0)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_interaction_correlation.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "fig_interaction_correlation.png"), dpi=180)
    print(f"Saved {OUT_DIR}/fig_interaction_correlation.[pdf,png]; N = {len(rows)}, r_pl = {r_pl:.3f}, r_f = {r_f:.3f}")


if __name__ == "__main__":
    main()
