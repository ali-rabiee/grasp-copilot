"""Inter-wizard agreement on the held-out subset.

Reports two κ scores per pair of wizards (and a Fleiss' κ when ≥3 wizards):

  1. Ask-vs-act κ — binary collapse: INTERACT vs {APPROACH, ALIGN_YAW}.
  2. Tool-selection κ given act — multi-class on (tool, target obj_id) for
     ticks where *both* wizards chose to act.

Inputs: a directory containing one ``grasp_gen.jsonl`` per wizard, all
annotated against the same held-out scenario set. The runner records each
tick's ``(episode_id, tick_idx)`` so alignment is unambiguous.

Run with::

    python -m wizard.analysis.kappa --agreement-dir <run_dir>/agreement
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def _load(path: Path) -> List[Tuple[Tuple[int, int], Dict]]:
    """Load (episode_id, tick_idx) → tool_call from one wizard's jsonl pair.

    Uses ``grasp_gen.jsonl`` aligned with the corresponding ``episodes_meta.jsonl``.
    """
    train_path = path / "grasp_gen.jsonl"
    meta_path = path / "episodes_meta.jsonl"
    if not train_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Need grasp_gen.jsonl + episodes_meta.jsonl in {path}")

    decisions: List[Tuple[Tuple[int, int], Dict]] = []
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta = json.loads(line)
            ep = int(meta["episode_id"])
            for d in meta.get("decisions", []):
                decisions.append(((ep, int(d["tick_idx"])), d["tool_call"]))
    return decisions


def _collapse_ask_act(tool_call: Dict) -> str:
    return "ASK" if tool_call.get("tool") == "INTERACT" else "ACT"


def _act_label(tool_call: Dict) -> str:
    tool = tool_call.get("tool")
    args = tool_call.get("args") or {}
    return f"{tool}::{args.get('obj','')}"


def cohen_kappa(labels_a: List[str], labels_b: List[str]) -> float:
    if len(labels_a) != len(labels_b) or not labels_a:
        return float("nan")
    n = len(labels_a)
    cats = sorted(set(labels_a) | set(labels_b))
    agree = sum(1 for x, y in zip(labels_a, labels_b) if x == y)
    p_o = agree / n
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    p_e = sum((counts_a[c] / n) * (counts_b[c] / n) for c in cats)
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else float("nan")
    return (p_o - p_e) / (1.0 - p_e)


def fleiss_kappa(matrix: List[List[int]]) -> float:
    """``matrix[i]`` is per-category counts at item i, summing to n_raters per row."""
    if not matrix:
        return float("nan")
    n = sum(matrix[0])
    N = len(matrix)
    if n < 2:
        return float("nan")
    k = len(matrix[0])
    p_j = [sum(matrix[i][j] for i in range(N)) / (N * n) for j in range(k)]
    P_i = [(sum(matrix[i][j] ** 2 for j in range(k)) - n) / (n * (n - 1)) for i in range(N)]
    P_bar = sum(P_i) / N
    P_e = sum(p ** 2 for p in p_j)
    if P_e == 1.0:
        return 1.0 if P_bar == 1.0 else float("nan")
    return (P_bar - P_e) / (1.0 - P_e)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agreement-dir", required=True,
                        help="Dir with one subdir per wizard, each holding "
                             "grasp_gen.jsonl + episodes_meta.jsonl")
    parser.add_argument("--out", default=None,
                        help="Optional path to write a kappa_report.txt")
    args = parser.parse_args()

    base = Path(args.agreement_dir)
    wizard_dirs = sorted([p for p in base.iterdir() if p.is_dir()])
    if len(wizard_dirs) < 2:
        raise SystemExit("Need ≥2 wizard subdirectories.")

    wiz_decisions: Dict[str, Dict[Tuple[int, int], Dict]] = {}
    for d in wizard_dirs:
        wiz_decisions[d.name] = dict(_load(d))

    # Aligned key set: items where every wizard provided a decision.
    common_keys = set.intersection(*[set(v.keys()) for v in wiz_decisions.values()])
    common_keys = sorted(common_keys)
    if not common_keys:
        raise SystemExit("No (episode_id, tick_idx) pairs are common across wizards.")

    lines: List[str] = []
    lines.append(f"Wizards: {[d.name for d in wizard_dirs]}")
    lines.append(f"Common decision points: {len(common_keys)}")
    lines.append("")

    # Pairwise Cohen κ (both axes).
    lines.append("=== Pairwise Cohen's κ ===")
    for a, b in itertools.combinations(wizard_dirs, 2):
        la_ask = [_collapse_ask_act(wiz_decisions[a.name][k]) for k in common_keys]
        lb_ask = [_collapse_ask_act(wiz_decisions[b.name][k]) for k in common_keys]
        k_ask = cohen_kappa(la_ask, lb_ask)

        act_keys = [k for k in common_keys
                    if _collapse_ask_act(wiz_decisions[a.name][k]) == "ACT"
                    and _collapse_ask_act(wiz_decisions[b.name][k]) == "ACT"]
        if act_keys:
            la_act = [_act_label(wiz_decisions[a.name][k]) for k in act_keys]
            lb_act = [_act_label(wiz_decisions[b.name][k]) for k in act_keys]
            k_act = cohen_kappa(la_act, lb_act)
        else:
            k_act = float("nan")
        lines.append(f"  {a.name} vs {b.name}: ask-vs-act κ = {k_ask:.3f}   "
                     f"tool-selection κ = {k_act:.3f}  (n_act={len(act_keys)})")
    lines.append("")

    # Fleiss κ for ask-vs-act when ≥3 wizards.
    if len(wizard_dirs) >= 3:
        cats = ["ASK", "ACT"]
        matrix: List[List[int]] = []
        for k in common_keys:
            row = [0, 0]
            for d in wizard_dirs:
                lab = _collapse_ask_act(wiz_decisions[d.name][k])
                row[cats.index(lab)] += 1
            matrix.append(row)
        lines.append(f"Fleiss' κ (ask-vs-act, {len(wizard_dirs)} raters): {fleiss_kappa(matrix):.3f}")

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n")


if __name__ == "__main__":
    main()
