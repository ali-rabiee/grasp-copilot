#!/usr/bin/env python3
"""
Publication-quality figures for the PRIME offline executive benchmark.

Generates 8 figures from evaluation outputs:
  Fig 1 – Radar chart: multi-metric model profiles
  Fig 2 – Grouped bar: main performance metrics
  Fig 3 – Stacked bar: error decomposition by failure type
  Fig 4 – Annotated heatmap: tool accuracy by dialog context
  Fig 5 – Grid (2×3): per-model tool confusion matrices
  Fig 6 – Grouped bar: tool accuracy by manipulation mode
  Fig 7 – Line chart: tool accuracy vs number of candidate objects
  Fig 8 – Scatter: accuracy–throughput trade-off

Usage:
    python make_offline_exec_figures.py \\
        --run_dir ../eval_outputs/paper_benchmark_run001 \\
        --out_dir . --tag paper_benchmark_run001
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np

# ── Palette & ordering ──────────────────────────────────────────────────

MODEL_ORDER = [
    "Qwen2.5-7B-FT", "Qwen2.5-3B-FT",
    "Qwen2.5-7B-ZS", "Qwen2.5-3B-ZS",
    "H1_ask_if_ambiguous", "H2_always_ask",
]

MODEL_COLORS = {
    "Qwen2.5-7B-FT":      "#2166AC",
    "Qwen2.5-3B-FT":      "#67A9CF",
    "Qwen2.5-7B-ZS":      "#B2182B",
    "Qwen2.5-3B-ZS":      "#EF8A62",
    "H1_ask_if_ambiguous": "#7B3294",
    "H2_always_ask":       "#C2A5CF",
}

PRETTY = {
    "Qwen2.5-7B-FT":      "Qwen 2.5-7B (FT)",
    "Qwen2.5-3B-FT":      "Qwen 2.5-3B (FT)",
    "Qwen2.5-7B-ZS":      "Qwen 2.5-7B (ZS)",
    "Qwen2.5-3B-ZS":      "Qwen 2.5-3B (ZS)",
    "H1_ask_if_ambiguous": "H\u2081: Ask-if-Ambiguous",
    "H2_always_ask":       "H\u2082: Always-Ask",
}

MODEL_MARKERS = {
    "Qwen2.5-7B-FT": "o", "Qwen2.5-3B-FT": "o",
    "Qwen2.5-7B-ZS": "s", "Qwen2.5-3B-ZS": "s",
    "H1_ask_if_ambiguous": "D", "H2_always_ask": "D",
}

METRIC_COLORS = ["#2166AC", "#4393C3", "#EF8A62", "#92C5DE"]

ERROR_COLORS = {
    "Correct":        "#2CA02C",
    "Wrong Tool":     "#E15759",
    "Schema Invalid": "#F28E2B",
    "JSON Invalid":   "#9467BD",
}

CONFUSION_CMAPS = {
    "Qwen2.5-7B-FT":      "Blues",
    "Qwen2.5-3B-FT":      "Purples",
    "Qwen2.5-7B-ZS":      "Reds",
    "Qwen2.5-3B-ZS":      "Oranges",
    "H1_ask_if_ambiguous": "Greens",
    "H2_always_ask":       "GnBu",
}


def _p(name: str) -> str:
    return PRETTY.get(name, name)

def _c(name: str) -> str:
    return MODEL_COLORS.get(name, "#888")

def _m(name: str) -> str:
    return MODEL_MARKERS.get(name, "o")

def _sk(name: str) -> int:
    try:
        return MODEL_ORDER.index(name)
    except ValueError:
        return 99


# ── Data loading ────────────────────────────────────────────────────────

def load_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def load_json_data(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_confusion(path: Path) -> Dict[str, Dict[str, Dict[str, int]]]:
    out: Dict[str, Dict[str, Dict[str, int]]] = {}
    cur: str | None = None
    header: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("==="):
                cur = line.strip("= ")
                out[cur] = {}
                continue
            if cur is None:
                continue
            parts = [p.strip() for p in line.split(",")]
            if parts[0] == "gt\\pred":
                header = parts[1:]
                continue
            gt = parts[0]
            out[cur][gt] = {}
            for h, c in zip(header, parts[1:]):
                try:
                    out[cur][gt][h] = int(c)
                except ValueError:
                    out[cur][gt][h] = 0
    return out


# ── Style ───────────────────────────────────────────────────────────────

def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })


def _save(fig: plt.Figure, out_dir: Path, tag: str, name: str) -> Path:
    p = out_dir / f"{tag}_{name}.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  \u2713 {p.name}")
    return p


# ── Figure 1 – Radar chart ─────────────────────────────────────────────

def fig_1_radar(summaries: list, out_dir: Path, tag: str) -> List[Path]:
    metrics = [
        ("tool_accuracy",          "Tool\nAccuracy"),
        ("strict_exact_rate",      "Strict Exact\nMatch"),
        ("motion_obj_accuracy",    "Motion-Obj\nAccuracy"),
        ("interact_kind_accuracy", "Interact-Kind\nAccuracy"),
        ("schema_valid_rate",      "Schema\nValid Rate"),
    ]

    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 7.5), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_rlabel_position(30)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"],
                       fontsize=7.5, color="#777")
    ax.set_ylim(0, 1.08)

    ax.yaxis.grid(True, color="#ccc", linewidth=0.6)
    ax.xaxis.grid(True, color="#ccc", linewidth=0.6)
    ax.spines["polar"].set_visible(False)

    for s in summaries:
        name = s["model"]["name"]
        values = [s.get(m, 0) for m, _ in metrics]
        values += values[:1]
        ax.plot(angles, values, "-", linewidth=2.2, label=_p(name),
                color=_c(name), marker="o", markersize=6,
                markerfacecolor=_c(name), markeredgecolor="white",
                markeredgewidth=1.0)
        ax.fill(angles, values, alpha=0.07, color=_c(name))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([lbl for _, lbl in metrics],
                       fontsize=10, fontweight="medium")

    fig.suptitle("Multi-Metric Model Profiles", fontsize=14,
                 fontweight="bold", y=0.98)

    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="lower center", fontsize=9, ncol=3,
        columnspacing=1.2, handlelength=1.8,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    return [_save(fig, out_dir, tag, "fig1_radar")]


# ── Figure 2 – Composite: (a) main metrics + (b) error decomposition ──

def fig_2_composite(csv_rows: list, out_dir: Path, tag: str) -> List[Path]:
    """Side-by-side composite: (a) grouped bar metrics, (b) error stacked bar."""
    from matplotlib.colors import to_rgb

    rows = sorted(csv_rows, key=lambda r: _sk(r["name"]))

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(14, 4.8),
        gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.35},
    )

    # ── Panel (a): Main performance metrics ─────────────────────
    metrics = [
        ("tool_accuracy",          "Tool Accuracy"),
        ("strict_exact_rate",      "Strict Exact Match"),
        ("motion_obj_accuracy",    "Motion-Obj Acc"),
        ("interact_kind_accuracy", "Interact-Kind Acc"),
    ]

    n_models = len(rows)
    n_met = len(metrics)
    x = np.arange(n_models)
    width = 0.19

    for i, (key, label) in enumerate(metrics):
        vals = [float(r.get(key, 0)) for r in rows]
        offset = (i - (n_met - 1) / 2) * width
        bars = ax_a.bar(x + offset, vals, width, label=label,
                        color=METRIC_COLORS[i], edgecolor="white",
                        linewidth=0.5, zorder=3)

        mc_rgb = to_rgb(METRIC_COLORS[i])
        lum = 0.299 * mc_rgb[0] + 0.587 * mc_rgb[1] + 0.114 * mc_rgb[2]
        inside_color = "white" if lum < 0.55 else "#222"

        for bar, v in zip(bars, vals):
            bx = bar.get_x() + bar.get_width() / 2
            if v >= 0.15:
                ax_a.text(
                    bx, bar.get_height() / 2,
                    f"{v:.1%}", ha="center", va="center",
                    rotation=90, fontsize=5.5, fontweight="bold",
                    color=inside_color, zorder=4,
                )
            else:
                ax_a.text(
                    bx, max(bar.get_height(), 0) + 0.01,
                    f"{v:.1%}", ha="center", va="bottom",
                    rotation=90, fontsize=5.5, fontweight="bold",
                    color="#444", zorder=4,
                )

    for sep in (1.5, 3.5):
        ax_a.axvline(sep, color="#ddd", linewidth=0.8, linestyle="--", zorder=1)

    ax_a.set_ylim(0, 1.16)
    ax_a.set_ylabel("Rate", fontweight="medium", fontsize=9)
    ax_a.set_xticks(x)
    names_compact = [
        "Qwen 2.5\n7B (FT)", "Qwen 2.5\n3B (FT)",
        "Qwen 2.5\n7B (ZS)", "Qwen 2.5\n3B (ZS)",
        "H\u2081: Ask-if\nAmbiguous", "H\u2082: Always\nAsk",
    ]
    ax_a.set_xticklabels(names_compact, rotation=0, ha="center", fontsize=7.5)
    ax_a.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax_a.grid(axis="y", alpha=0.25, zorder=0)
    ax_a.legend(loc="upper center", ncol=4, fontsize=6.5,
                framealpha=0.9, edgecolor="#eee", frameon=True,
                fancybox=True, handlelength=1.2, columnspacing=0.8,
                bbox_to_anchor=(0.5, 1.12))

    # ── Panel (b): Error decomposition ──────────────────────────
    rows_rev = rows[::-1]
    names_rev = [_p(r["name"]) for r in rows_rev]
    n_total = int(rows_rev[0]["n"])

    correct, wrong_tool, schema_inv, json_inv = [], [], [], []
    for r in rows_rev:
        n = int(r["n"])
        je = int(r.get("json_errors", 0))
        se = int(r.get("schema_errors", 0))
        tc = round(float(r["tool_accuracy"]) * n)
        wt = n - tc - je - se
        correct.append(tc)
        wrong_tool.append(max(wt, 0))
        schema_inv.append(se)
        json_inv.append(je)

    y = np.arange(len(names_rev))
    cats = [
        (correct,    ERROR_COLORS["Correct"],        "Correct"),
        (wrong_tool, ERROR_COLORS["Wrong Tool"],     "Wrong Tool"),
        (schema_inv, ERROR_COLORS["Schema Invalid"], "Schema Invalid"),
        (json_inv,   ERROR_COLORS["JSON Invalid"],   "JSON Invalid"),
    ]

    left = np.zeros(len(names_rev))
    for vals, color, label in cats:
        arr = np.array(vals, dtype=float)
        ax_b.barh(y, arr, left=left, height=0.62, color=color,
                  edgecolor="white", linewidth=0.4, label=label)
        left += arr

    for i, r in enumerate(rows_rev):
        pct = float(r["tool_accuracy"])
        ax_b.text(n_total + 20, i, f"{pct:.1%}", va="center", ha="left",
                  fontsize=8, fontweight="bold", color=_c(r["name"]))

    ax_b.set_yticks(y)
    ax_b.set_yticklabels(names_rev, fontsize=8)
    ax_b.set_xlim(0, n_total * 1.12)
    ax_b.legend(loc="upper center", ncol=4, fontsize=6.5,
                framealpha=0.9, edgecolor="#eee", frameon=True,
                fancybox=True, handlelength=1.2, columnspacing=0.8,
                bbox_to_anchor=(0.5, 1.12))
    ax_b.grid(axis="x", alpha=0.2)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)

    pos_a = ax_a.get_position()
    pos_b = ax_b.get_position()
    fig.text((pos_a.x0 + pos_a.x1) / 2, 0.025, "(a)",
             ha="center", fontsize=13, fontweight="bold")
    fig.text((pos_b.x0 + pos_b.x1) / 2, 0.025, "(b)",
             ha="center", fontsize=13, fontweight="bold")

    return [_save(fig, out_dir, tag, "fig2_performance_overview")]


# ── Figure 4 – Context heatmap ─────────────────────────────────────────

def fig_4_context_heatmap(ctx_rows: list, out_dir: Path,
                          tag: str) -> List[Path]:
    rows = sorted(ctx_rows, key=lambda r: _sk(r["model"]))

    ctx_spec = [
        ("no_context_acc",             "No Context",              "no_context_n"),
        ("intent_gate_candidates_acc", "Intent Gate\nCandidates", "intent_gate_candidates_n"),
        ("candidate_choice_acc",       "Candidate\nChoice",       "candidate_choice_n"),
        ("confirm_acc",                "Confirm",                 "confirm_n"),
        ("mode_select_acc",            "Mode\nSelect",            "mode_select_n"),
        ("intent_gate_yaw_acc",        "Intent Gate\nYaw",        "intent_gate_yaw_n"),
        ("anything_else_acc",          "Anything\nElse",          "anything_else_n"),
    ]

    data = np.array([[float(r[c]) for c, _, _ in ctx_spec] for r in rows])
    n_per_ctx = [int(rows[0].get(nc, 0)) for _, _, nc in ctx_spec]
    ylabels = [_p(r["model"]) for r in rows]
    xlabels = [f"{lbl}\n(n={n})" for _, lbl, _ in ctx_spec
               for n in [n_per_ctx[list(range(len(ctx_spec)))[
                   [x[1] for x in ctx_spec].index(lbl)]]]]

    # Rebuild xlabels cleanly
    xlabels = []
    for idx, (_, lbl, _) in enumerate(ctx_spec):
        xlabels.append(f"{lbl}\n(n={n_per_ctx[idx]})")

    cmap = LinearSegmentedColormap.from_list(
        "rdylgn_custom",
        ["#B2182B", "#EF8A62", "#FEE08B", "#D9EF8B", "#66BD63", "#1A9850"],
        N=256,
    )

    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(data, vmin=0.0, vmax=1.0, aspect="auto", cmap=cmap)

    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=9.5)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, fontsize=8.5, ha="center")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            color = "white" if val < 0.45 else "#222"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color=color)

    for edge in range(data.shape[0] + 1):
        ax.axhline(edge - 0.5, color="white", linewidth=2.5)
    for edge in range(data.shape[1] + 1):
        ax.axvline(edge - 0.5, color="white", linewidth=2.5)

    for sep in (1.5, 3.5):
        ax.axhline(sep, color="#444", linewidth=1.8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.02)
    cbar.set_label("Tool Accuracy", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    ax.set_title("Tool Accuracy by Dialog Context", pad=14)

    fig.tight_layout()
    return [_save(fig, out_dir, tag, "fig4_context_heatmap")]


# ── Figure 5 – Confusion matrix grid ───────────────────────────────────

def fig_5_confusion_grid(conf_data: dict, out_dir: Path,
                         tag: str) -> List[Path]:
    tools = ["APPROACH", "ALIGN_YAW", "INTERACT"]
    pred_labels = ["APPROACH", "ALIGN_YAW", "INTERACT", "INVALID"]
    model_names = [m for m in MODEL_ORDER if m in conf_data]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    fig.suptitle("Tool Confusion Matrices (Row-Normalised)", fontsize=14,
                 fontweight="bold", y=0.99)

    for idx, model in enumerate(model_names):
        ax = axes[idx // 3, idx % 3]
        m = conf_data[model]

        mat = np.zeros((3, 4))
        for i, gt in enumerate(tools):
            if gt not in m:
                continue
            for j, pr in enumerate(tools):
                mat[i, j] = m[gt].get(pr, 0)
            mat[i, 3] = (m[gt].get("INVALID_JSON", 0)
                         + m[gt].get("INVALID_SCHEMA", 0))

        row_sums = mat.sum(axis=1, keepdims=True)
        mat_norm = np.divide(mat, row_sums,
                             out=np.zeros_like(mat), where=row_sums > 0)

        cmap_name = CONFUSION_CMAPS.get(model, "Blues")
        im = ax.imshow(mat_norm, vmin=0, vmax=1, cmap=cmap_name,
                       aspect="auto")

        ax.set_xticks(range(4))
        ax.set_xticklabels(pred_labels, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(3))
        ax.set_yticklabels(tools, fontsize=8.5)
        ax.set_title(_p(model), fontsize=11, fontweight="bold",
                     color=_c(model), pad=8)

        for i in range(3):
            for j in range(4):
                pct = mat_norm[i, j]
                cnt = int(mat[i, j])
                txt_color = "white" if pct > 0.55 else "black"
                if cnt > 0:
                    ax.text(j, i, f"{pct:.0%}\n({cnt})", ha="center",
                            va="center", fontsize=7.5,
                            fontweight="bold" if pct > 0.5 else "normal",
                            color=txt_color)
                elif pct == 0:
                    ax.text(j, i, "0%", ha="center", va="center",
                            fontsize=7.5, color="#bbb")

        for edge in range(4):
            ax.axhline(edge - 0.5, color="white", linewidth=1.5)
        for edge in range(5):
            ax.axvline(edge - 0.5, color="white", linewidth=1.5)

        if idx // 3 == 1:
            ax.set_xlabel("Predicted", fontsize=9, fontweight="medium")
        if idx % 3 == 0:
            ax.set_ylabel("Ground Truth", fontsize=9, fontweight="medium")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return [_save(fig, out_dir, tag, "fig5_confusion_grid")]


# ── Figure 6 – Performance by manipulation mode ────────────────────────

def fig_6_mode_performance(summaries: list, out_dir: Path,
                           tag: str) -> List[Path]:
    modes = ["translation", "rotation", "gripper"]
    mode_labels = ["Translation", "Rotation", "Gripper"]

    n_modes = len(modes)
    n_models = len(summaries)
    x = np.arange(n_modes)
    width = 0.12

    fig, ax = plt.subplots(figsize=(10, 5.2))

    for i, s in enumerate(summaries):
        name = s["model"]["name"]
        vals = []
        for mode in modes:
            bm = s.get("by_mode", {}).get(mode, {})
            n = bm.get("n", 1)
            tc = bm.get("tool_correct", 0)
            vals.append(tc / n if n > 0 else 0)

        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=_p(name),
                      color=_c(name), edgecolor="white", linewidth=0.5,
                      zorder=3)
        for bar, v in zip(bars, vals):
            if v > 0.08:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.018,
                        f"{v:.0%}", ha="center", va="bottom",
                        fontsize=6, color="#444")

    mode_ns = [summaries[0]["by_mode"].get(m, {}).get("n", 0) for m in modes]
    for j, n in enumerate(mode_ns):
        ax.text(j, -0.08, f"n={n}", ha="center", va="top", fontsize=8.5,
                color="#888", transform=ax.get_xaxis_transform())

    ax.set_ylim(0, 1.16)
    ax.set_ylabel("Tool Accuracy", fontweight="medium")
    ax.set_title("Tool Accuracy by Manipulation Mode", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels, fontsize=10.5, fontweight="medium")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(loc="upper right", ncol=2, fontsize=8,
              framealpha=0.9, edgecolor="#eee", frameon=True, fancybox=True)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14)
    return [_save(fig, out_dir, tag, "fig6_mode_performance")]


# ── Figure 7 – Tool accuracy vs number of candidates ───────────────────

def fig_7_candidate_scaling(summaries: list, out_dir: Path,
                            tag: str) -> List[Path]:
    cand_keys = ["cands_0", "cands_1", "cands_2", "cands_3",
                 "cands_4", "cands_5", "cands_6+"]
    cand_labels = ["0", "1", "2", "3", "4", "5", "6+"]

    first_bc = summaries[0].get("by_num_candidates", {})
    cand_ns = [first_bc.get(k, {}).get("n", 0) for k in cand_keys]

    fig, ax = plt.subplots(figsize=(9, 5.2))

    for s in summaries:
        name = s["model"]["name"]
        bc = s.get("by_num_candidates", {})
        vals = []
        for k in cand_keys:
            entry = bc.get(k, {})
            n = entry.get("n", 0)
            tc = entry.get("tool_correct", 0)
            vals.append(tc / n if n > 0 else 0)

        ax.plot(range(len(vals)), vals, "-", linewidth=2.2, markersize=7,
                label=_p(name), color=_c(name), marker=_m(name),
                markerfacecolor=_c(name), markeredgecolor="white",
                markeredgewidth=1.0)

    ax.set_ylim(-0.02, 1.08)
    ax.set_xlim(-0.3, len(cand_keys) - 0.7)
    ax.set_ylabel("Tool Accuracy", fontweight="medium")
    ax.set_xlabel("Number of Candidate Objects", fontweight="medium")
    ax.set_title("Tool Accuracy vs. Number of Candidate Objects", pad=12)

    tick_labels = [f"{lbl}\n($n$={n})" for lbl, n in zip(cand_labels, cand_ns)]
    ax.set_xticks(range(len(tick_labels)))
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(loc="lower left", ncol=2, fontsize=8.5,
              framealpha=0.9, edgecolor="#eee", frameon=True, fancybox=True)

    fig.tight_layout()
    return [_save(fig, out_dir, tag, "fig7_candidate_scaling")]


# ── Figure 8 – Accuracy vs throughput ───────────────────────────────────

def fig_8_accuracy_throughput(csv_rows: list, out_dir: Path,
                              tag: str) -> List[Path]:
    rows = sorted(csv_rows, key=lambda r: _sk(r["name"]))

    fig, ax = plt.subplots(figsize=(9, 5.5))

    label_offsets = {
        "Qwen2.5-7B-FT":      (12, 10,  "left"),
        "Qwen2.5-3B-FT":      (12, -12, "left"),
        "Qwen2.5-7B-ZS":      (12, 8,   "left"),
        "Qwen2.5-3B-ZS":      (12, -12, "left"),
        "H1_ask_if_ambiguous": (-12, 12, "right"),
        "H2_always_ask":       (-12, -12, "right"),
    }

    for r in rows:
        name = r["name"]
        throughput = float(r.get("examples_per_sec", 0))
        acc = float(r.get("tool_accuracy", 0))
        strict = float(r.get("strict_exact_rate", 0))

        size = max(strict * 600 + 40, 55)

        ax.scatter([throughput], [acc], s=size, c=_c(name), marker=_m(name),
                   edgecolors="white", linewidths=1.5, zorder=5, alpha=0.92)

        ox, oy, ha = label_offsets.get(name, (10, 8, "left"))
        ax.annotate(
            _p(name), (throughput, acc),
            textcoords="offset points", xytext=(ox, oy),
            fontsize=8.5, fontweight="medium", color=_c(name), ha=ha,
            arrowprops=dict(arrowstyle="-", color="#ccc", lw=0.7),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Throughput (examples / sec)  [log scale]",
                  fontweight="medium")
    ax.set_ylabel("Tool Accuracy", fontweight="medium")
    ax.set_title("Accuracy vs. Throughput Trade-off", pad=12)
    ax.set_ylim(0.05, 1.08)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(alpha=0.2, zorder=0)

    ax.text(0.02, 0.02, "Bubble size \u221d strict exact match rate",
            transform=ax.transAxes, fontsize=8, color="#999", style="italic")

    type_legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2166AC",
               markersize=9, label="Fine-Tuned"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#B2182B",
               markersize=9, label="Zero-Shot"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#7B3294",
               markersize=9, label="Heuristic"),
    ]
    ax.legend(handles=type_legend, loc="center left", fontsize=9,
              framealpha=0.9, edgecolor="#eee", frameon=True, fancybox=True)

    fig.tight_layout()
    return [_save(fig, out_dir, tag, "fig8_accuracy_throughput")]


# ── Main ────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate publication figures")
    ap.add_argument(
        "--run_dir", type=str,
        default=str(Path(__file__).resolve().parent.parent
                    / "eval_outputs" / "paper_benchmark_run001"),
    )
    ap.add_argument(
        "--out_dir", type=str,
        default=str(Path(__file__).resolve().parent),
    )
    ap.add_argument("--tag", type=str, default="paper_benchmark_run001")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = load_csv(run_dir / "summary_all.csv")
    context_csv = load_csv(run_dir / "context_breakdown.csv")
    conf_data = parse_confusion(run_dir / "confusion_matrices.csv")
    json_data = load_json_data(run_dir / "summary_all.json")

    summaries = json_data["summaries"]
    summaries = sorted(summaries, key=lambda s: _sk(s["model"]["name"]))

    setup_style()

    print(f"\nGenerating figures \u2192 {out_dir}/\n")

    outs: List[Path] = []
    outs += fig_1_radar(summaries, out_dir, args.tag)
    outs += fig_2_composite(summary_csv, out_dir, args.tag)
    outs += fig_4_context_heatmap(context_csv, out_dir, args.tag)
    outs += fig_5_confusion_grid(conf_data, out_dir, args.tag)
    outs += fig_6_mode_performance(summaries, out_dir, args.tag)
    outs += fig_7_candidate_scaling(summaries, out_dir, args.tag)
    outs += fig_8_accuracy_throughput(summary_csv, out_dir, args.tag)

    print(f"\nDone \u2013 {len(outs)} figures written.\n")


if __name__ == "__main__":
    main()
