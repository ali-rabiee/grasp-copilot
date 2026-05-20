"""
Build conference-paper tables from paper_benchmark outputs.

Reads:
    evaluation/results/paper_benchmark/summary_all.csv
    evaluation/results/paper_benchmark/per_model_results/*.json   (for richer metrics)

Writes:
    evaluation/results/paper_benchmark/tables/
        table_1_main.csv / .tex             # per-env tool accuracy + macro avg
        table_2_ambiguous.csv / .tex        # ambiguous vs clean
        table_3_ablations.csv / .tex        # per-env LoRA, rank, warm-start
        table_full_metrics.csv              # full flat table (every metric)

LaTeX style: booktabs, bolded best per column within a table section.

Usage:
    python -m evaluation.tables.build_paper_tables
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCH_DIR = REPO_ROOT / "evaluation" / "results" / "paper_benchmark"


# =============================================================================
# Display configuration
# =============================================================================

# Trained-model groups (top of Table I).
MAIN_MODEL_ORDER = [
    "oracle_lora",
    "woz_lora",
    "oracle_woz_lora",
]
# Ablations (shown in a separate panel / table).
ABLATION_ORDER = ["oracle_woz_r32", "oracle_ycb", "oracle_stacking", "oracle_pouring"]
# Baselines (bottom of Table I).
BASELINE_ORDER = ["h1_ask_if_amb", "h2_always_ask", "sa1_pred_assist", "sa2_bayes_intent"]
# Zero-shot (optional).
ZS_ORDER = ["qwen3b_zs"]


ENV_COLS = ["ycb", "stacking", "pouring"]
ENV_DISPLAY = {"ycb": "YCB", "stacking": "Stack", "pouring": "Pour"}


# =============================================================================
# Loading
# =============================================================================

def _load_cells(bench_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for jp in sorted((bench_dir / "results").glob("*.json")):
        try:
            with open(jp, "r", encoding="utf-8") as f:
                rows.append(json.load(f))
        except Exception as e:
            print(f"[warn] failed to read {jp}: {e}")
    return rows


def _index_by(rows: List[Dict[str, Any]], model_key: str = "_model_safe", set_key: str = "_eval_set"):
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in rows:
        k = (str(r.get(model_key)), str(r.get(set_key)))
        out[k] = r
    return out


# =============================================================================
# Number formatting
# =============================================================================

def _pct(x: Optional[float], nan_str: str = "--") -> str:
    if x is None:
        return nan_str
    try:
        v = float(x)
        if v != v:  # NaN
            return nan_str
        return f"{v*100:.1f}"
    except Exception:
        return nan_str


def _fmt_bold(values: List[Optional[float]], higher_is_better: bool = True) -> List[str]:
    """Return formatted percentage strings; bold the best one."""
    finite = [(i, v) for i, v in enumerate(values) if isinstance(v, (int, float))]
    if not finite:
        return [_pct(v) for v in values]
    best_idx = max(finite, key=lambda kv: kv[1])[0] if higher_is_better else min(finite, key=lambda kv: kv[1])[0]
    out: List[str] = []
    for i, v in enumerate(values):
        s = _pct(v)
        out.append(f"\\textbf{{{s}}}" if i == best_idx and s != "--" else s)
    return out


# =============================================================================
# Generic LaTeX writers
# =============================================================================

def _write_latex_table(
    path: Path,
    *,
    caption: str,
    label: str,
    col_specs: str,
    header: List[str],
    body_rows: List[List[str]],
    rule_after: Optional[List[int]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rule_after = rule_after or []
    lines: List[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_specs}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for i, row in enumerate(body_rows):
        lines.append(" & ".join(row) + " \\\\")
        if i in rule_after and i < len(body_rows) - 1:
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, header: List[str], rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


# =============================================================================
# Table I — Per-env headline
# =============================================================================

def _gather_per_env(cells_idx, model_safe: str, flavor: str) -> Dict[str, Optional[float]]:
    """For a given model, fetch tool_acc per env for the requested flavor."""
    # Set name convention: <flavor>_valid_<env> or ambiguous_<env>.
    if flavor == "oracle":
        names = {e: f"oracle_valid_{e}" for e in ENV_COLS}
    elif flavor == "ambiguous":
        # We renamed reach_to_grasp_ycb -> ycb in the runner registry; check the actual file name.
        names = {
            "ycb":      "ambiguous_ycb",
            "stacking": "ambiguous_stacking",
            "pouring":  "ambiguous_pouring",
        }
    else:
        raise ValueError(flavor)

    out: Dict[str, Optional[float]] = {}
    for env, set_name in names.items():
        cell = cells_idx.get((model_safe, set_name))
        out[env] = cell.get("tool_accuracy") if cell else None
    return out


def _macro_avg(values: Dict[str, Optional[float]]) -> Optional[float]:
    nums = [v for v in values.values() if isinstance(v, (int, float))]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _display_for(rows, model_safe: str) -> str:
    for r in rows:
        if r.get("_model_safe") == model_safe:
            return r.get("_display") or model_safe
    return model_safe


def build_table_1_main(rows: List[Dict[str, Any]], out_dir: Path, *, include_zs: bool) -> None:
    """Per-env tool accuracy on oracle valid + WoZ valid, with macro-3 average."""
    idx = _index_by(rows)

    section_order: List[Tuple[str, List[str]]] = [
        ("Main models",  [m for m in MAIN_MODEL_ORDER if any(k[0] == m for k in idx)]),
    ]
    if include_zs and any(k[0] in ZS_ORDER for k in idx):
        section_order.append(("Zero-shot", [m for m in ZS_ORDER if any(k[0] == m for k in idx)]))
    section_order.append(("Baselines", [m for m in BASELINE_ORDER if any(k[0] == m for k in idx)]))

    # Build value matrix
    matrix: List[Tuple[str, List[Optional[float]]]] = []
    rule_after: List[int] = []
    flat_rows: List[List[str]] = []
    for si, (sec_label, model_keys) in enumerate(section_order):
        for mk in model_keys:
            per_env = _gather_per_env(idx, mk, "oracle")
            woz_cell = idx.get((mk, "woz_valid"))
            woz_acc = woz_cell.get("tool_accuracy") if woz_cell else None
            avg3 = _macro_avg(per_env)
            row_vals = [per_env["ycb"], per_env["stacking"], per_env["pouring"], avg3, woz_acc]
            matrix.append((mk, row_vals))
        # Insert section rule (after last row of this section)
        if si < len(section_order) - 1 and model_keys:
            rule_after.append(len(matrix) - 1)

    # Bold best per column within each section
    body: List[List[str]] = []
    # Find best per column per section to bold
    section_bounds: List[Tuple[int, int]] = []
    cur = 0
    for sec_label, model_keys in section_order:
        if model_keys:
            section_bounds.append((cur, cur + len(model_keys)))
            cur += len(model_keys)

    # Build cells with bolding (per column, bold the overall best across all rows)
    cols = len(matrix[0][1]) if matrix else 0
    col_strs: List[List[str]] = [[] for _ in range(cols)]
    for c in range(cols):
        col_values = [row[1][c] for row in matrix]
        col_strs[c] = _fmt_bold(col_values, higher_is_better=True)

    for ri, (mk, _) in enumerate(matrix):
        body.append([_display_for(rows, mk)] + [col_strs[c][ri] for c in range(cols)])
        flat_rows.append([_display_for(rows, mk), mk] + [_pct(matrix[ri][1][c]) for c in range(cols)])

    header_disp = ["Model", "YCB", "Stack", "Pour", "Avg-3", "WoZ"]
    # Plain (un-bolded) CSV from `flat_rows` so it's machine-readable.
    _write_csv(out_dir / "table_1_main.csv", header_disp,
               [[fr[0]] + fr[2:] for fr in flat_rows])
    _write_latex_table(
        out_dir / "table_1_main.tex",
        caption=(
            "Tool-call accuracy (\\%) by environment on held-out validation sets. "
            "Trained 3B LoRAs (top) outperform heuristic shared-autonomy baselines "
            "(bottom) across all environments. The headline model is Oracle$\\rightarrow$WoZ-LoRA."
        ),
        label="tab:per_env",
        col_specs="l" + "c" * len(header_disp[1:]),
        header=header_disp,
        body_rows=body,
        rule_after=rule_after,
    )


# =============================================================================
# Table II — Ambiguous vs Clean
# =============================================================================

def build_table_2_ambiguous(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    """For each model in {oracle, woz, oracle_woz} × baselines, show clean (Avg-3 on oracle valid)
    vs ambiguous (Avg-3 on ambiguous sets), and the gap."""
    idx = _index_by(rows)
    targets = [m for m in MAIN_MODEL_ORDER if any(k[0] == m for k in idx)]
    targets += [m for m in BASELINE_ORDER if any(k[0] == m for k in idx)]

    body: List[List[str]] = []
    csv_rows: List[List[str]] = []
    clean_vals: List[Optional[float]] = []
    amb_vals: List[Optional[float]] = []
    gap_vals: List[Optional[float]] = []
    for mk in targets:
        per_env_clean = _gather_per_env(idx, mk, "oracle")
        per_env_amb   = _gather_per_env(idx, mk, "ambiguous")
        clean = _macro_avg(per_env_clean)
        amb = _macro_avg(per_env_amb)
        gap = (amb - clean) if (isinstance(amb, (int, float)) and isinstance(clean, (int, float))) else None
        clean_vals.append(clean); amb_vals.append(amb); gap_vals.append(gap)
    clean_b = _fmt_bold(clean_vals, higher_is_better=True)
    amb_b   = _fmt_bold(amb_vals,   higher_is_better=True)
    # Gap: smallest absolute drop is best (closest to 0 from below). Bold smallest |drop|.
    gap_b: List[str] = []
    finite_gaps = [(i, v) for i, v in enumerate(gap_vals) if isinstance(v, (int, float))]
    best_gap_idx = min(finite_gaps, key=lambda kv: abs(kv[1]))[0] if finite_gaps else -1
    for i, v in enumerate(gap_vals):
        if not isinstance(v, (int, float)):
            gap_b.append("--")
            continue
        sign = "+" if v >= 0 else ""
        s = f"{sign}{v*100:.1f}"
        gap_b.append(f"\\textbf{{{s}}}" if i == best_gap_idx else s)

    # Find the index of the last "main model" present in targets — that's where
    # we insert the midrule before baselines.
    main_present = [m for m in MAIN_MODEL_ORDER if m in targets]
    last_main = main_present[-1] if main_present else None
    rule_after: List[int] = []
    for i, mk in enumerate(targets):
        if mk == last_main and i < len(targets) - 1:
            rule_after.append(i)
        body.append([_display_for(rows, mk), clean_b[i], amb_b[i], gap_b[i]])
        csv_rows.append([_display_for(rows, mk), _pct(clean_vals[i]), _pct(amb_vals[i]),
                         f"{(gap_vals[i] or 0)*100:+.1f}" if isinstance(gap_vals[i], (int, float)) else "--"])

    header = ["Model", "Clean (Avg-3)", "Ambiguous (Avg-3)", "$\\Delta$ (pp)"]
    _write_csv(out_dir / "table_2_ambiguous.csv",
               ["Model", "Clean_Avg3", "Ambiguous_Avg3", "Delta_pp"],
               csv_rows)
    _write_latex_table(
        out_dir / "table_2_ambiguous.tex",
        caption=(
            "Tool-call accuracy (\\%) on clean vs.\\ ambiguous scenarios, "
            "macro-averaged across YCB / Stack / Pour. $\\Delta$ is the absolute "
            "percentage-point change from clean to ambiguous; closer to zero is better. "
            "WoZ supervision narrows the gap that heuristic baselines suffer."
        ),
        label="tab:ambiguous_vs_clean",
        col_specs="l" + "c" * (len(header) - 1),
        header=header,
        body_rows=body,
        rule_after=rule_after,
    )


# =============================================================================
# Table III — Ablations
# =============================================================================

def build_table_3_ablations(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    """Three ablation panels: per-env vs unified, LoRA rank, warm-start."""
    idx = _index_by(rows)

    panels: List[Tuple[str, List[str]]] = [
        ("Warm-start (Phase-2 base)", [m for m in ["woz_lora", "oracle_lora", "oracle_woz_lora"] if any(k[0] == m for k in idx)]),
        ("LoRA rank",                 [m for m in ["oracle_woz_lora", "oracle_woz_r32"] if any(k[0] == m for k in idx)]),
        ("Per-env vs.\\ unified",     [m for m in ["oracle_lora", "oracle_ycb", "oracle_stacking", "oracle_pouring"] if any(k[0] == m for k in idx)]),
    ]

    body: List[List[str]] = []
    csv_rows: List[List[str]] = []
    rule_after: List[int] = []
    section_header_rows: List[int] = []
    for pi, (label, model_keys) in enumerate(panels):
        # Insert a sub-header row spanning all columns.
        section_header_rows.append(len(body))
        body.append(["\\multicolumn{6}{l}{\\emph{" + label + "}}"] + [""] * 5)
        clean_label = label.replace("\\\\", "").replace("\\ ", " ")
        csv_rows.append(["# " + clean_label] + [""] * 5)

        col_values: Dict[str, List[Optional[float]]] = {e: [] for e in ENV_COLS}
        avg_values: List[Optional[float]] = []
        amb_values: List[Optional[float]] = []
        for mk in model_keys:
            per_env = _gather_per_env(idx, mk, "oracle")
            for e in ENV_COLS:
                col_values[e].append(per_env[e])
            avg_values.append(_macro_avg(per_env))
            per_amb = _gather_per_env(idx, mk, "ambiguous")
            amb_values.append(_macro_avg(per_amb))

        bold_per_env = {e: _fmt_bold(col_values[e]) for e in ENV_COLS}
        bold_avg = _fmt_bold(avg_values)
        bold_amb = _fmt_bold(amb_values)

        for ri, mk in enumerate(model_keys):
            body.append([
                _display_for(rows, mk),
                bold_per_env["ycb"][ri],
                bold_per_env["stacking"][ri],
                bold_per_env["pouring"][ri],
                bold_avg[ri],
                bold_amb[ri],
            ])
            csv_rows.append([
                _display_for(rows, mk),
                _pct(col_values["ycb"][ri]),
                _pct(col_values["stacking"][ri]),
                _pct(col_values["pouring"][ri]),
                _pct(avg_values[ri]),
                _pct(amb_values[ri]),
            ])

        if pi < len(panels) - 1:
            rule_after.append(len(body) - 1)

    header = ["Variant", "YCB", "Stack", "Pour", "Avg-3", "Ambiguous"]
    _write_csv(out_dir / "table_3_ablations.csv",
               ["Variant", "YCB", "Stack", "Pour", "Avg3", "Ambiguous_Avg3"],
               csv_rows)
    _write_latex_table(
        out_dir / "table_3_ablations.tex",
        caption=(
            "Ablation study of the three design axes: (i) the warm-start choice "
            "for Phase-2 supervision, (ii) the LoRA rank, and (iii) one unified "
            "model vs.\\ per-environment models. All numbers are tool-call accuracy (\\%)."
        ),
        label="tab:ablations",
        col_specs="l" + "c" * (len(header) - 1),
        header=header,
        body_rows=body,
        rule_after=rule_after,
    )


# =============================================================================
# Full per-cell CSV
# =============================================================================

def build_full_metrics_csv(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    headers = [
        "model_safe", "model_display", "group",
        "eval_set", "env", "flavor",
        "n", "tool_accuracy",
        "schema_valid_rate", "strict_exact_rate",
        "motion_obj_accuracy", "interact_kind_accuracy",
        "examples_per_sec",
    ]
    out = []
    for r in rows:
        timing = r.get("timing") or {}
        out.append([
            r.get("_model_safe"), r.get("_display"), r.get("_group"),
            r.get("_eval_set"), r.get("_env"), r.get("_flavor"),
            r.get("n"),
            f"{r.get('tool_accuracy', 0):.4f}",
            f"{r.get('schema_valid_rate', 0):.4f}",
            f"{r.get('strict_exact_rate', 0):.4f}",
            f"{r.get('motion_obj_accuracy', 0):.4f}",
            f"{r.get('interact_kind_accuracy', 0):.4f}",
            f"{timing.get('examples_per_sec', 0):.2f}",
        ])
    _write_csv(out_dir / "table_full_metrics.csv", headers, out)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench_dir", default=str(DEFAULT_BENCH_DIR))
    ap.add_argument("--out_dir", default=None, help="Default: <bench_dir>/tables")
    ap.add_argument("--include_zs", action="store_true", help="Include zero-shot reference row in Table I")
    args = ap.parse_args()

    bench_dir = Path(args.bench_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (bench_dir / "tables")

    rows = _load_cells(bench_dir)
    if not rows:
        raise SystemExit(f"No result cells found under {bench_dir}/results — run run_paper_benchmark first.")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[tables] loaded {len(rows)} cells from {bench_dir}")
    print(f"[tables] writing to {out_dir}")

    build_table_1_main(rows, out_dir, include_zs=args.include_zs)
    build_table_2_ambiguous(rows, out_dir)
    build_table_3_ablations(rows, out_dir)
    build_full_metrics_csv(rows, out_dir)

    for f in sorted(out_dir.iterdir()):
        print(f"  - {f.relative_to(bench_dir)}")


if __name__ == "__main__":
    main()
