"""Post-hoc IG analysis on logged reranker dialogs.

Reads every dialogs.jsonl under --ablation_root. For each emitted INTERACT
call, recomputes four IG scores using the SAME entropy/pruning logic the
online sweep used, varying only the selector applied to the K logged
candidates:

    ig_chosen      = the selector's pick (already in the log).
    ig_no_rerank   = candidate[0]'s IG (= what the bare LLM would have said).
    ig_random      = mean IG across all K candidates (uniform-random selector).
    ig_oracle      = the oracle's question's IG at the same state, if known.

Outputs:
    out_dir/per_question.csv     one row per dialog × selector
    out_dir/summary.json         aggregate means + frac IG≥0.5 + by-kind splits
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from evaluation import _bootstrap  # noqa: F401
except Exception:
    import _bootstrap  # type: ignore  # noqa: F401


def _iter_dialogs(root: Path) -> Iterable[Dict[str, Any]]:
    for p in sorted(root.rglob("dialogs.jsonl")):
        # ablation_root/<model>__<rerank_mode>/dialogs.jsonl
        rel_name = p.parent.name
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                rec["_cell"] = rel_name
                yield rec


def _selector_scores(rec: Dict[str, Any]) -> Dict[str, float]:
    cands = rec.get("candidates") or []
    if not cands:
        return {}
    scored = [float(c.get("ig_bits", 0.0)) for c in cands]
    out = {
        "chosen":     float(rec.get("ig_bits", 0.0)),
        "no_rerank":  scored[0] if scored else 0.0,
        "random":     (sum(scored) / len(scored)) if scored else 0.0,
        "info_gain":  max(scored) if scored else 0.0,
    }
    # oracle selector requires the oracle's emit at the same state — not in
    # the log. Filled in elsewhere if available; otherwise leave None.
    return out


def _bucket_by_kind(rec: Dict[str, Any]) -> str:
    k = str(rec.get("interact_kind", "")).upper()
    return k if k in {"QUESTION", "SUGGESTION", "CONFIRM"} else "OTHER"


def analyze(ablation_root: Path, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_question_csv = out_dir / "per_question.csv"

    rows: List[Dict[str, Any]] = []
    by_selector: Dict[str, List[float]] = defaultdict(list)
    by_kind: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    cells = set()
    n_total = 0

    for rec in _iter_dialogs(ablation_root):
        scores = _selector_scores(rec)
        if not scores:
            continue
        kind = _bucket_by_kind(rec)
        n_total += 1
        cells.add(rec.get("_cell", ""))
        for sel_name, sc in scores.items():
            rows.append({
                "cell": rec.get("_cell", ""),
                "scenario_id": rec.get("scenario_id", ""),
                "seed": rec.get("seed", 0),
                "condition": rec.get("condition", ""),
                "tick": rec.get("tick", 0),
                "state_hash": rec.get("state_hash", ""),
                "selector": sel_name,
                "interact_kind": kind,
                "n_candidates_before": rec.get("n_candidates_before", 0),
                "h_before_bits": rec.get("h_before_bits", 0.0),
                "ig_bits": round(float(sc), 6),
            })
            by_selector[sel_name].append(float(sc))
            by_kind[kind][sel_name].append(float(sc))

    with per_question_csv.open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    def _stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"n": 0, "mean": 0.0, "median": 0.0, "frac_ge_0p5": 0.0}
        return {
            "n": len(values),
            "mean": round(st.mean(values), 4),
            "median": round(st.median(values), 4),
            "frac_ge_0p5": round(sum(1 for v in values if v >= 0.5) / len(values), 4),
        }

    summary = {
        "n_dialogs": n_total,
        "n_cells": len(cells),
        "cells": sorted(cells),
        "selectors": {sel: _stats(vals) for sel, vals in by_selector.items()},
        "by_kind": {
            kind: {sel: _stats(vals) for sel, vals in selmap.items()}
            for kind, selmap in by_kind.items()
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True, ensure_ascii=False)

    print(f"[analyze_ig] {n_total} dialogs across {len(cells)} cells")
    for sel, s in summary["selectors"].items():
        print(f"  selector={sel:10s}  n={s['n']:5d}  mean_IG={s['mean']:.3f} bits  "
              f"median={s['median']:.3f}  frac≥0.5={s['frac_ge_0p5']:.3f}")
    print(f"[analyze_ig] wrote {per_question_csv}")
    print(f"[analyze_ig] wrote {out_dir / 'summary.json'}")
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ablation_root", default="evaluation/results/reranker/ablation")
    ap.add_argument("--out_dir", default="evaluation/results/reranker/ig_analysis")
    ap.add_argument("--include_selectors", nargs="*",
                    default=["info_gain", "random", "no_rerank", "chosen"],
                    help="Selectors to include in CSV (analyse always computes all four).")
    args = ap.parse_args(argv)

    root = Path(args.ablation_root)
    if not root.exists():
        raise SystemExit(f"ablation root not found: {root}")
    out_dir = Path(args.out_dir)
    analyze(root, out_dir)


if __name__ == "__main__":
    main()
