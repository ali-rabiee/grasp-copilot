from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _set_rcparams() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def _model_order(name: str) -> Tuple[int, str]:
    # Put FT first, then ZS, then heuristics.
    if "7B-FT" in name:
        return (0, name)
    if "3B-FT" in name:
        return (1, name)
    if "7B-ZS" in name:
        return (2, name)
    if "3B-ZS" in name:
        return (3, name)
    if name.startswith("H1"):
        return (4, name)
    if name.startswith("H2"):
        return (5, name)
    return (9, name)


def _pretty_name(name: str) -> str:
    return (
        name.replace("Qwen2.5-", "Qwen2.5 ")
        .replace("-FT", " (FT)")
        .replace("-ZS", " (ZS)")
        .replace("H1_ask_if_ambiguous", "H1: ask-if-ambig")
        .replace("H2_always_ask", "H2: always-ask")
    )


def fig_main_metrics(summary_csv: Path, out_dir: Path, tag: str) -> List[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = _read_csv(summary_csv)
    rows = sorted(rows, key=lambda r: _model_order(r["name"]))

    names = [_pretty_name(r["name"]) for r in rows]
    tool = [_safe_float(r["tool_accuracy"]) for r in rows]
    motion_obj = [_safe_float(r["motion_obj_accuracy"]) for r in rows]
    interact_kind = [_safe_float(r["interact_kind_accuracy"]) for r in rows]
    schema = [_safe_float(r["schema_valid_rate"]) for r in rows]

    x = np.arange(len(names))
    w = 0.2

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(x - 1.5 * w, tool, w, label="Tool acc")
    ax.bar(x - 0.5 * w, motion_obj, w, label="Motion obj acc")
    ax.bar(x + 0.5 * w, interact_kind, w, label="Interact kind acc")
    ax.bar(x + 1.5 * w, schema, w, label="Schema-valid rate")

    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Rate")
    ax.set_title("Offline executive benchmark (main metrics)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower left", ncols=2, frameon=False)
    fig.tight_layout()

    p = out_dir / f"{tag}_fig1_main_metrics.png"
    fig.savefig(p)
    plt.close(fig)
    return [p]


def fig_validity(summary_csv: Path, out_dir: Path, tag: str) -> List[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = _read_csv(summary_csv)
    rows = sorted(rows, key=lambda r: _model_order(r["name"]))
    names = [_pretty_name(r["name"]) for r in rows]

    json_valid = [_safe_float(r["json_valid_rate"]) for r in rows]
    schema_valid = [_safe_float(r["schema_valid_rate"]) for r in rows]
    json_err = [1.0 - v for v in json_valid]
    schema_err = [1.0 - v for v in schema_valid]

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.bar(x - w / 2, json_err, w, label="Invalid JSON rate")
    ax.bar(x + w / 2, schema_err, w, label="Invalid schema rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Rate")
    ax.set_title("Output reliability")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    p = out_dir / f"{tag}_fig2_reliability_invalid_rates.png"
    fig.savefig(p)
    plt.close(fig)
    return [p]


def fig_tradeoff(summary_csv: Path, out_dir: Path, tag: str) -> List[Path]:
    import matplotlib.pyplot as plt

    rows = _read_csv(summary_csv)
    rows = sorted(rows, key=lambda r: _model_order(r["name"]))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for r in rows:
        name = _pretty_name(r["name"])
        x = _safe_float(r.get("examples_per_sec", "0"))
        y = _safe_float(r.get("tool_accuracy", "0"))
        ax.scatter([x], [y], s=70)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 5), fontsize=9)

    ax.set_xlabel("Examples / second")
    ax.set_ylabel("Tool accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Accuracy vs throughput (hardware-dependent)")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    p = out_dir / f"{tag}_fig3_tradeoff_accuracy_vs_speed.png"
    fig.savefig(p)
    plt.close(fig)
    return [p]


def fig_context_heatmap(context_csv: Path, out_dir: Path, tag: str) -> List[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = _read_csv(context_csv)
    rows = sorted(rows, key=lambda r: _model_order(r["model"]))

    # Columns: *_acc
    cols = [c for c in rows[0].keys() if c.endswith("_acc")]
    # Keep a stable, meaningful ordering if present:
    preferred = [
        "no_context_acc",
        "intent_gate_candidates_acc",
        "candidate_choice_acc",
        "confirm_acc",
        "mode_select_acc",
        "intent_gate_yaw_acc",
        "anything_else_acc",
    ]
    cols_sorted = [c for c in preferred if c in cols] + [c for c in cols if c not in preferred]

    data = np.array([[float(r[c]) for c in cols_sorted] for r in rows], dtype=float)
    ylabels = [_pretty_name(r["model"]) for r in rows]
    xlabels = [c.replace("_acc", "") for c in cols_sorted]

    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    im = ax.imshow(data, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=25, ha="right")
    ax.set_title("Tool accuracy by dialog context")

    # annotate
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Accuracy")
    fig.tight_layout()

    p = out_dir / f"{tag}_fig4_context_heatmap.png"
    fig.savefig(p)
    plt.close(fig)
    return [p]


def _parse_confusion_csv(path: Path) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Parse confusion_matrices.csv written by evaluation code.
    Returns: model -> gt_tool -> pred_tool -> count
    """
    out: Dict[str, Dict[str, Dict[str, int]]] = {}
    cur_model: str | None = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("===") and line.endswith("==="):
                cur_model = line.strip("=").strip()
                out[cur_model] = {}
                continue
            if cur_model is None:
                continue
            parts = [p.strip() for p in line.split(",")]
            if parts[0] == "gt\\pred":
                header = parts[1:]
                continue
            gt = parts[0]
            counts = parts[1:]
            row = out[cur_model].setdefault(gt, {})
            for h, c in zip(header, counts):
                try:
                    row[h] = int(c)
                except Exception:
                    row[h] = 0
    return out


def fig_confusion(conf_csv: Path, out_dir: Path, tag: str) -> List[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    conf = _parse_confusion_csv(conf_csv)
    # Focus on 7B FT vs 7B ZS if present, else first two models.
    candidates = []
    for k in conf.keys():
        if "7B-FT" in k:
            candidates.insert(0, k)
        elif "7B-ZS" in k:
            candidates.append(k)
    if len(candidates) < 2:
        candidates = list(conf.keys())[:2]
    tools = ["APPROACH", "ALIGN_YAW", "INTERACT"]

    out_paths: List[Path] = []
    for model_name in candidates[:2]:
        m = conf.get(model_name, {})
        mat = np.array([[m.get(gt, {}).get(pr, 0) for pr in tools] for gt in tools], dtype=float)
        # normalize rows
        row_sum = mat.sum(axis=1, keepdims=True)
        mat_norm = np.divide(mat, row_sum, out=np.zeros_like(mat), where=row_sum > 0)

        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        im = ax.imshow(mat_norm, vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(tools)))
        ax.set_xticklabels(tools, rotation=25, ha="right")
        ax.set_yticks(range(len(tools)))
        ax.set_yticklabels(tools)
        ax.set_title(f"Tool confusion (row-normalized)\n{_pretty_name(model_name)}")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{mat_norm[i, j]:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03).set_label("P(pred | gt)")
        fig.tight_layout()

        p = out_dir / f"{tag}_fig5_confusion_{model_name.replace('/', '_')}.png"
        fig.savefig(p)
        plt.close(fig)
        out_paths.append(p)

    return out_paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dir",
        type=str,
        default="/home/ali/github/ali-rabiee/grasp-copilot/evaluation/eval_outputs/paper_benchmark_run001",
        help="Path to eval_outputs/<run>/ directory (contains summary_all.csv, context_breakdown.csv, confusion_matrices.csv).",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="/home/ali/github/ali-rabiee/grasp-copilot/evaluation/plots",
        help="Where to write figures.",
    )
    ap.add_argument("--tag", type=str, default="paper_benchmark_run001", help="Prefix for output filenames.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    _ensure_out_dir(out_dir)

    summary_csv = run_dir / "summary_all.csv"
    context_csv = run_dir / "context_breakdown.csv"
    conf_csv = run_dir / "confusion_matrices.csv"

    for p in (summary_csv, context_csv, conf_csv):
        if not p.exists():
            raise SystemExit(f"Missing required file: {p}")

    _set_rcparams()

    outs: List[Path] = []
    outs += fig_main_metrics(summary_csv, out_dir, args.tag)
    outs += fig_validity(summary_csv, out_dir, args.tag)
    outs += fig_tradeoff(summary_csv, out_dir, args.tag)
    outs += fig_context_heatmap(context_csv, out_dir, args.tag)
    outs += fig_confusion(conf_csv, out_dir, args.tag)

    print("[make_offline_exec_figures] wrote:")
    for p in outs:
        print(" -", p)


if __name__ == "__main__":
    main()


