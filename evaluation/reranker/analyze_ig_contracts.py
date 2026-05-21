"""Appendix robustness check: IG measured on held-out contract eval sets.

For every contract row whose GT tool_call is INTERACT, compute the
information gain of that question using the SAME pruning + entropy
modules the online sweep uses. This is a single-system measurement (no
reranker comparison — there's no LLM in the loop here), used purely as
a sensitivity check that the scenario-sweep IG number isn't an artifact
of the scenario corpus.

Inputs (defaults match run_paper_benchmark.py):
    data/woz_phase2/llm_contract_valid.jsonl                  → "WoZ valid"
    data/oracle_valid_ycb/llm_contract_200.jsonl              → "Oracle YCB"
    data/oracle_valid_stacking/llm_contract_200.jsonl         → "Oracle Stack"
    data/oracle_valid_pouring/llm_contract_200.jsonl          → "Oracle Pour"

Outputs (default → evaluation/results/reranker/ig_analysis_contracts/):
    per_question_contracts.csv   one row per INTERACT
    summary_contracts.json       per-set + overall stats
    appendix_ig_contracts.{csv,tex}  appendix table
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from evaluation import _bootstrap  # noqa: F401
except Exception:
    import _bootstrap  # type: ignore  # noqa: F401

from llm.reranker.entropy import information_gain_bits
from llm.reranker.pruning import PruneSnapshot, infer_pruning_intent


@dataclass(frozen=True)
class ContractSet:
    name: str          # short id
    display: str       # paper label
    path: str          # relative to repo root


DEFAULT_SETS: List[ContractSet] = [
    ContractSet("woz_valid",             "WoZ valid",     "data/woz_phase2/llm_contract_valid.jsonl"),
    ContractSet("oracle_valid_ycb",      "Oracle YCB",    "data/oracle_valid_ycb/llm_contract_200.jsonl"),
    ContractSet("oracle_valid_stacking", "Oracle Stack",  "data/oracle_valid_stacking/llm_contract_200.jsonl"),
    ContractSet("oracle_valid_pouring",  "Oracle Pour",   "data/oracle_valid_pouring/llm_contract_200.jsonl"),
]


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _score_one(input_str: str, output_str: str, prior: str) -> Optional[Dict[str, Any]]:
    try:
        inp = json.loads(input_str)
        gt = json.loads(output_str)
    except Exception:
        return None
    if not isinstance(gt, dict) or gt.get("tool") != "INTERACT":
        return None

    mem = inp.get("memory") or {}
    objs = inp.get("objects") or []
    gh = inp.get("gripper_hist") or []
    snap = PruneSnapshot.from_memory(mem)
    intent = infer_pruning_intent(gt, snap.candidates, objs)
    n_ch = len(((gt.get("args") or {}).get("choices") or []))
    ig, h_b, h_a, _ = information_gain_bits(
        intent, n_ch, snap,
        prior=prior, gripper_hist=gh, objects=objs,
    )
    kind = str((gt.get("args") or {}).get("kind", "")).upper()
    return {
        "interact_kind": kind if kind in {"QUESTION", "CONFIRM", "SUGGESTION"} else "OTHER",
        "context_type_hint": intent.kind,
        "n_candidates_before": len(snap.candidates),
        "h_before_bits": round(float(h_b), 6),
        "h_after_expected_bits": round(float(h_a), 6),
        "ig_bits": round(float(ig), 6),
    }


def _stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "frac_ge_0p5": 0.0}
    return {
        "n": len(values),
        "mean": round(st.mean(values), 4),
        "median": round(st.median(values), 4),
        "frac_ge_0p5": round(sum(1 for v in values if v >= 0.5) / len(values), 4),
    }


def run(sets: List[ContractSet], repo_root: Path, out_dir: Path, prior: str) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_q_path = out_dir / "per_question_contracts.csv"
    per_q_rows: List[Dict[str, Any]] = []
    per_set: Dict[str, List[float]] = defaultdict(list)
    per_set_kind: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for s in sets:
        p = (repo_root / s.path) if not Path(s.path).is_absolute() else Path(s.path)
        if not p.exists():
            print(f"[contracts] skip {s.name}: not found ({p})")
            continue
        n_int = 0
        for row in _iter_jsonl(p):
            rec = _score_one(str(row.get("input", "")), str(row.get("output", "")), prior)
            if rec is None:
                continue
            n_int += 1
            rec["set"] = s.name
            rec["set_display"] = s.display
            per_q_rows.append(rec)
            per_set[s.name].append(rec["ig_bits"])
            per_set_kind[s.name][rec["interact_kind"]].append(rec["ig_bits"])
        print(f"[contracts] {s.name}: scored {n_int} INTERACT rows from {p.name}")

    with per_q_path.open("w", newline="") as f:
        if per_q_rows:
            fields = ["set", "set_display", "interact_kind", "context_type_hint",
                      "n_candidates_before", "h_before_bits", "h_after_expected_bits", "ig_bits"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(per_q_rows)

    summary = {
        "prior": prior,
        "sets": {
            name: {
                "display": next((s.display for s in sets if s.name == name), name),
                "overall": _stats(vals),
                "by_kind": {k: _stats(v) for k, v in per_set_kind[name].items()},
            }
            for name, vals in per_set.items()
        },
        "overall": _stats([v for vals in per_set.values() for v in vals]),
    }
    (out_dir / "summary_contracts.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False)
    )

    # Appendix table.
    csv_path = out_dir / "appendix_ig_contracts.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["set", "display", "n", "mean_ig", "median_ig", "frac_ge_0p5"])
        w.writeheader()
        for name, s in summary["sets"].items():
            ov = s["overall"]
            w.writerow({
                "set": name, "display": s["display"], "n": ov["n"],
                "mean_ig": ov["mean"], "median_ig": ov["median"], "frac_ge_0p5": ov["frac_ge_0p5"],
            })
        ov = summary["overall"]
        w.writerow({
            "set": "ALL", "display": "Pooled", "n": ov["n"],
            "mean_ig": ov["mean"], "median_ig": ov["median"], "frac_ge_0p5": ov["frac_ge_0p5"],
        })

    tex_path = out_dir / "appendix_ig_contracts.tex"
    with tex_path.open("w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Sensitivity check: information gain of held-out INTERACT calls "
                "measured on contract eval sets (no LLM in the loop). Confirms the "
                "scenario-sweep IG number is not an artefact of the scenario corpus.}\n")
        f.write("\\label{tab:ig_contracts_appendix}\n")
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        f.write("Eval set & N & Mean IG (b) & Median IG (b) & Frac IG$\\geq$0.5 \\\\\n\\midrule\n")
        for _, s in summary["sets"].items():
            ov = s["overall"]
            f.write(f"{s['display']} & {ov['n']} & {ov['mean']:.3f} & "
                    f"{ov['median']:.3f} & {ov['frac_ge_0p5']:.3f} \\\\\n")
        f.write("\\midrule\n")
        ov = summary["overall"]
        f.write(f"\\textbf{{Pooled}} & {ov['n']} & \\textbf{{{ov['mean']:.3f}}} & "
                f"{ov['median']:.3f} & {ov['frac_ge_0p5']:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"[contracts] wrote {per_q_path}")
    print(f"[contracts] wrote {out_dir / 'summary_contracts.json'}")
    print(f"[contracts] wrote {csv_path}")
    print(f"[contracts] wrote {tex_path}")
    for name, s in summary["sets"].items():
        ov = s["overall"]
        print(f"  {s['display']:18s}  n={ov['n']:4d}  mean_IG={ov['mean']:.3f}  "
              f"frac≥0.5={ov['frac_ge_0p5']:.3f}")
    ov = summary["overall"]
    print(f"  {'POOLED':18s}  n={ov['n']:4d}  mean_IG={ov['mean']:.3f}  "
          f"frac≥0.5={ov['frac_ge_0p5']:.3f}")
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default="evaluation/results/reranker/ig_analysis_contracts")
    ap.add_argument("--prior", default="uniform", choices=["uniform", "motion_weighted"])
    ap.add_argument("--sets", default=None,
                    help="Comma-separated subset of "
                         "{woz_valid,oracle_valid_ycb,oracle_valid_stacking,oracle_valid_pouring}")
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir

    sets = DEFAULT_SETS
    if args.sets:
        keep = {x.strip() for x in args.sets.split(",") if x.strip()}
        sets = [s for s in sets if s.name in keep]
        if not sets:
            raise SystemExit(f"no eval sets selected after filter: {args.sets}")

    run(sets, repo_root, out_dir, args.prior)


if __name__ == "__main__":
    main()
