#!/usr/bin/env python3
"""
Publication-quality figures for the PRIME 3-env benchmark.

Generates from the paper_benchmark output directory:
    fig1_per_env_bars.{pdf,png}      Tool accuracy by env, grouped by model
    fig2_ambiguous_gap.{pdf,png}     Clean vs ambiguous accuracy per model
    fig3_radar.{pdf,png}             Multi-metric model profile
    fig4_confusion_grid.{pdf,png}    Per-model tool confusion matrices
    fig5_context_heatmap.{pdf,png}   Tool accuracy by dialog-context type
    fig6_accuracy_throughput.{pdf,png}  Trade-off scatter
    fig7_error_breakdown.{pdf,png}   Stacked error decomposition

Usage:
    python -m evaluation.plots.paper_figures
    python -m evaluation.plots.paper_figures --eval_set oracle_valid_ycb
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCH_DIR = REPO_ROOT / "evaluation" / "results" / "paper_benchmark"


# =============================================================================
# Style
# =============================================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Stable model ordering used by every figure.
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

MODEL_COLORS = {
    "oracle_lora":       "#67A9CF",
    "woz_lora":          "#F4A582",
    "oracle_woz_lora":   "#2166AC",
    "oracle_woz_r32":    "#1A5490",
    "oracle_ycb":        "#92C5DE",
    "oracle_stacking":   "#FDDBC7",
    "oracle_pouring":    "#D6604D",
    "qwen3b_zs":         "#999999",
    "h1_ask_if_amb":     "#7B3294",
    "h2_always_ask":     "#C2A5CF",
    "sa1_pred_assist":   "#4DAF4A",
    "sa2_bayes_intent":  "#984EA3",
}

ENV_ORDER = ["ycb", "stacking", "pouring"]
ENV_LABEL = {"ycb": "YCB", "stacking": "Stack", "pouring": "Pour"}


# =============================================================================
# Loading
# =============================================================================

def load_cells(bench_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for jp in sorted((bench_dir / "results").glob("*.json")):
        try:
            with open(jp, "r", encoding="utf-8") as f:
                rows.append(json.load(f))
        except Exception:
            continue
    return rows


def index_cells(cells: List[Dict[str, Any]]):
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for c in cells:
        out[(str(c.get("_model_safe")), str(c.get("_eval_set")))] = c
    return out


def display_of(cells: List[Dict[str, Any]], model_safe: str) -> str:
    for c in cells:
        if c.get("_model_safe") == model_safe:
            return c.get("_display") or model_safe
    return model_safe


def models_present(cells: List[Dict[str, Any]]) -> List[str]:
    seen = {c.get("_model_safe") for c in cells}
    return [m for m in MODEL_ORDER if m in seen]


# =============================================================================
# Fig 1 — Per-env grouped bars
# =============================================================================

def fig1_per_env_bars(cells: List[Dict[str, Any]], out_dir: Path) -> None:
    idx = index_cells(cells)
    models = [m for m in models_present(cells)
              if m in {"oracle_lora", "woz_lora", "oracle_woz_lora",
                       "h1_ask_if_amb", "h2_always_ask", "sa1_pred_assist", "sa2_bayes_intent"}]
    if not models:
        return

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    n_envs = len(ENV_ORDER)
    n_models = len(models)
    bar_w = 0.85 / n_models
    x = np.arange(n_envs)

    for mi, m in enumerate(models):
        vals = []
        for env in ENV_ORDER:
            cell = idx.get((m, f"oracle_valid_{env}"))
            vals.append(cell.get("tool_accuracy", 0) * 100 if cell else 0)
        offset = (mi - n_models / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, vals, width=bar_w,
                      color=MODEL_COLORS.get(m, "#666"),
                      edgecolor="black", linewidth=0.4,
                      label=display_of(cells, m))
        for b, v in zip(bars, vals):
            if v > 2:
                ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}",
                        ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([ENV_LABEL[e] for e in ENV_ORDER])
    ax.set_ylabel("Tool-call accuracy (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Tool-call accuracy by environment (oracle held-out validation)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=min(n_models, 4),
              frameon=False, fontsize=8.5)
    _save(fig, out_dir / "fig1_per_env_bars")


# =============================================================================
# Fig 2 — Ambiguous vs Clean
# =============================================================================

def fig2_ambiguous_gap(cells: List[Dict[str, Any]], out_dir: Path) -> None:
    idx = index_cells(cells)
    targets = [m for m in models_present(cells)
               if m in {"oracle_lora", "woz_lora", "oracle_woz_lora",
                        "h1_ask_if_amb", "h2_always_ask",
                        "sa1_pred_assist", "sa2_bayes_intent"}]
    if not targets:
        return

    def macro_avg(model: str, flavor_prefix: str) -> float:
        vals = []
        for env in ENV_ORDER:
            cell = idx.get((model, f"{flavor_prefix}_{env}"))
            if cell:
                vals.append(cell.get("tool_accuracy", 0))
        return float(np.mean(vals)) * 100 if vals else 0.0

    clean = [macro_avg(m, "oracle_valid") for m in targets]
    amb = [macro_avg(m, "ambiguous") for m in targets]

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(targets))
    w = 0.38
    bars1 = ax.bar(x - w/2, clean, width=w, color="#67A9CF", edgecolor="black", linewidth=0.4, label="Clean (Avg-3)")
    bars2 = ax.bar(x + w/2, amb,   width=w, color="#D6604D", edgecolor="black", linewidth=0.4, label="Ambiguous (Avg-3)")
    for b, v in zip(bars1, clean):
        ax.text(b.get_x() + b.get_width()/2, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)
    for b, v in zip(bars2, amb):
        ax.text(b.get_x() + b.get_width()/2, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)

    # Connector arrows showing the gap
    for xi, (c, a) in enumerate(zip(clean, amb)):
        ax.annotate("", xy=(xi + w/2, a), xytext=(xi - w/2, c),
                    arrowprops=dict(arrowstyle="-|>", color="0.3", lw=0.8, alpha=0.45))

    ax.set_xticks(x)
    ax.set_xticklabels([display_of(cells, m) for m in targets], rotation=18, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Tool-call accuracy (%)")
    ax.set_title("Clean vs. ambiguous scenarios — WoZ supervision narrows the gap")
    ax.legend(loc="upper right", frameon=True)
    _save(fig, out_dir / "fig2_ambiguous_gap")


# =============================================================================
# Fig 3 — Radar
# =============================================================================

RADAR_METRICS = [
    ("Tool acc.",       "tool_accuracy"),
    ("Motion obj.",     "motion_obj_accuracy"),
    ("Interact kind",   "interact_kind_accuracy"),
    ("Schema valid",    "schema_valid_rate"),
    ("Strict match",    "strict_exact_rate"),
]


def fig3_radar(cells: List[Dict[str, Any]], eval_set: str, out_dir: Path) -> None:
    idx = index_cells(cells)
    targets = [m for m in models_present(cells)
               if m in {"oracle_lora", "woz_lora", "oracle_woz_lora",
                        "h1_ask_if_amb", "h2_always_ask"}]
    if not targets:
        return

    labels = [m for m, _ in RADAR_METRICS]
    keys = [k for _, k in RADAR_METRICS]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5.2, 5.2), subplot_kw=dict(polar=True))
    for m in targets:
        cell = idx.get((m, eval_set))
        if cell is None:
            continue
        vals = [float(cell.get(k, 0)) for k in keys]
        vals += vals[:1]
        ax.plot(angles, vals, "-o", color=MODEL_COLORS.get(m, "#666"),
                lw=1.6, ms=4.0, label=display_of(cells, m))
        ax.fill(angles, vals, alpha=0.10, color=MODEL_COLORS.get(m, "#666"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Multi-metric profile — {eval_set}", y=1.10)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    _save(fig, out_dir / "fig3_radar")


# =============================================================================
# Fig 4 — Confusion-matrix grid
# =============================================================================

TOOLS_ALL = ["INTERACT", "APPROACH", "ALIGN_YAW", "STACK", "GRAB", "POUR"]


def fig4_confusion_grid(cells: List[Dict[str, Any]], eval_set: str, out_dir: Path) -> None:
    idx = index_cells(cells)
    targets = [m for m in models_present(cells)
               if m in {"oracle_lora", "woz_lora", "oracle_woz_lora",
                        "h1_ask_if_amb", "h2_always_ask", "sa2_bayes_intent"}]
    if not targets:
        return
    n = len(targets)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows),
                              constrained_layout=True)
    axes = np.atleast_2d(axes)

    for i, m in enumerate(targets):
        ax = axes[i // cols][i % cols]
        cell = idx.get((m, eval_set))
        if not cell:
            ax.axis("off")
            continue
        cm = cell.get("tool_confusion") or {}
        # build matrix over tools present in either axis
        used_gt = [t for t in TOOLS_ALL if t in cm]
        used_pr = sorted({pr for gt in used_gt for pr in cm.get(gt, {})})
        # focus on real tools (ignore INVALID_* sinks for the row-normalized plot)
        pr_tools = [t for t in used_pr if t in TOOLS_ALL]
        if not used_gt or not pr_tools:
            ax.axis("off")
            continue
        mat = np.zeros((len(used_gt), len(pr_tools)))
        for r, gt in enumerate(used_gt):
            row = cm.get(gt, {})
            total = sum(row.get(p, 0) for p in pr_tools) or 1
            for c, p in enumerate(pr_tools):
                mat[r, c] = row.get(p, 0) / total
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(pr_tools)))
        ax.set_xticklabels(pr_tools, rotation=35, ha="right", fontsize=7)
        ax.set_yticks(range(len(used_gt)))
        ax.set_yticklabels(used_gt, fontsize=7)
        ax.set_title(display_of(cells, m), fontsize=9)
        # annotate cells with values
        for r in range(len(used_gt)):
            for c in range(len(pr_tools)):
                v = mat[r, c]
                if v >= 0.05:
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                            color="white" if v > 0.55 else "black", fontsize=6.5)
        ax.set_xlabel("predicted", fontsize=8)
        ax.set_ylabel("ground truth", fontsize=8)

    for j in range(len(targets), rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle(f"Tool confusion (row-normalized) — {eval_set}")
    _save(fig, out_dir / "fig4_confusion_grid")


# =============================================================================
# Fig 5 — Per-context heatmap
# =============================================================================

def fig5_context_heatmap(cells: List[Dict[str, Any]], eval_set: str, out_dir: Path) -> None:
    idx = index_cells(cells)
    targets = [m for m in models_present(cells)
               if m in {"oracle_lora", "woz_lora", "oracle_woz_lora",
                        "h1_ask_if_amb", "h2_always_ask",
                        "sa1_pred_assist", "sa2_bayes_intent"}]
    if not targets:
        return
    # Discover contexts
    all_ctx = set()
    for m in targets:
        cell = idx.get((m, eval_set))
        if cell:
            all_ctx.update((cell.get("by_context") or {}).keys())
    contexts = sorted(c for c in all_ctx if c not in {"invalid_input_json"})
    if not contexts:
        return
    mat = np.full((len(targets), len(contexts)), np.nan)
    n_mat = np.zeros_like(mat)
    for ri, m in enumerate(targets):
        cell = idx.get((m, eval_set))
        if not cell:
            continue
        by_ctx = cell.get("by_context") or {}
        for ci, ctx in enumerate(contexts):
            d = by_ctx.get(ctx, {})
            n = d.get("n", 0)
            if n:
                mat[ri, ci] = d.get("tool_correct", 0) / n
                n_mat[ri, ci] = n

    fig, ax = plt.subplots(figsize=(max(7.0, 0.6 * len(contexts)), 0.5 * len(targets) + 1.5))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(contexts)))
    ax.set_xticklabels(contexts, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels([display_of(cells, m) for m in targets], fontsize=9)
    for ri in range(len(targets)):
        for ci in range(len(contexts)):
            v = mat[ri, ci]
            if not np.isnan(v):
                ax.text(ci, ri, f"{v*100:.0f}", ha="center", va="center",
                        color="white" if v > 0.55 else "black", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.020, pad=0.02, label="Tool accuracy")
    ax.set_title(f"Tool accuracy by dialog context — {eval_set}")
    _save(fig, out_dir / "fig5_context_heatmap")


# =============================================================================
# Fig 6 — Accuracy vs throughput
# =============================================================================

def fig6_acc_throughput(cells: List[Dict[str, Any]], eval_set: str, out_dir: Path) -> None:
    idx = index_cells(cells)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for m in models_present(cells):
        cell = idx.get((m, eval_set))
        if not cell:
            continue
        timing = cell.get("timing") or {}
        thr = float(timing.get("examples_per_sec", 0))
        acc = float(cell.get("tool_accuracy", 0)) * 100
        if thr <= 0:
            continue
        ax.scatter(thr, acc, s=120, color=MODEL_COLORS.get(m, "#666"),
                   edgecolors="black", linewidth=0.6, label=display_of(cells, m), zorder=3)
        ax.annotate(display_of(cells, m), (thr, acc), xytext=(6, 4),
                    textcoords="offset points", fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("Throughput (examples / second, log scale)")
    ax.set_ylabel("Tool accuracy (%)")
    ax.set_title(f"Accuracy–throughput trade-off — {eval_set}")
    ax.set_ylim(0, 105)
    _save(fig, out_dir / "fig6_accuracy_throughput")


# =============================================================================
# Fig 7 — Error breakdown stacked
# =============================================================================

def fig7_error_breakdown(cells: List[Dict[str, Any]], eval_set: str, out_dir: Path) -> None:
    idx = index_cells(cells)
    targets = [m for m in models_present(cells)
               if m in {"oracle_lora", "woz_lora", "oracle_woz_lora",
                        "qwen3b_zs", "h1_ask_if_amb", "h2_always_ask"}]
    if not targets:
        return
    fractions = []
    for m in targets:
        cell = idx.get((m, eval_set))
        n = cell.get("n", 0) if cell else 0
        if not n:
            fractions.append((0, 0, 0, 0))
            continue
        correct = cell.get("tool_accuracy", 0)
        # Json invalid / schema invalid from confusion matrix totals.
        cm = cell.get("tool_confusion") or {}
        n_invalid_json = sum(row.get("INVALID_JSON", 0) for row in cm.values())
        n_invalid_schema = sum(row.get("INVALID_SCHEMA", 0) for row in cm.values())
        invalid_json_frac = n_invalid_json / n
        invalid_schema_frac = n_invalid_schema / n
        wrong_tool_frac = max(0, 1.0 - correct - invalid_json_frac - invalid_schema_frac)
        fractions.append((correct, wrong_tool_frac, invalid_schema_frac, invalid_json_frac))

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    arr = np.array(fractions) * 100  # to %
    labels = ["Correct", "Wrong tool", "Schema invalid", "JSON invalid"]
    colors = ["#2CA02C", "#E15759", "#F28E2B", "#9467BD"]
    bottom = np.zeros(len(targets))
    for i, lab in enumerate(labels):
        ax.barh(range(len(targets)), arr[:, i], left=bottom, color=colors[i],
                edgecolor="black", linewidth=0.4, label=lab)
        bottom += arr[:, i]
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels([display_of(cells, m) for m in targets])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of predictions (%)")
    ax.set_title(f"Outcome decomposition — {eval_set}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)
    _save(fig, out_dir / "fig7_error_breakdown")


# =============================================================================
# Save helper
# =============================================================================

def _save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(stem) + ".pdf")
    fig.savefig(str(stem) + ".png", dpi=200)
    plt.close(fig)
    print(f"  - {stem.name}.{{pdf,png}}")


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench_dir", default=str(DEFAULT_BENCH_DIR))
    ap.add_argument("--out_dir", default=None, help="Default: <bench_dir>/figures")
    ap.add_argument("--eval_set", default="oracle_valid_ycb",
                    help="Eval set used by single-set figures (radar, confusion, context, throughput, errors)")
    args = ap.parse_args()

    bench_dir = Path(args.bench_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (bench_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = load_cells(bench_dir)
    if not cells:
        raise SystemExit(f"No cells under {bench_dir}/results — run run_paper_benchmark first.")
    print(f"[figs] loaded {len(cells)} cells")

    fig1_per_env_bars(cells, out_dir)
    fig2_ambiguous_gap(cells, out_dir)
    fig3_radar(cells, args.eval_set, out_dir)
    fig4_confusion_grid(cells, args.eval_set, out_dir)
    fig5_context_heatmap(cells, args.eval_set, out_dir)
    fig6_acc_throughput(cells, args.eval_set, out_dir)
    fig7_error_breakdown(cells, args.eval_set, out_dir)


if __name__ == "__main__":
    main()
