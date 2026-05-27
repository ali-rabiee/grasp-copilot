#!/usr/bin/env python3
"""Rewrite benchmark summary CSVs from cached per-cell result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from evaluation.benchmarks.run_paper_benchmark import (
    EVAL_SETS,
    HEURISTIC_MODELS,
    TRAINED_MODELS,
    ZERO_SHOT_MODEL,
    ZERO_SHOT_STRONG_MODEL,
    EvalSet,
    ModelEntry,
    _write_manifest,
)


def _known_models() -> Dict[str, ModelEntry]:
    models = list(TRAINED_MODELS) + [ZERO_SHOT_MODEL, ZERO_SHOT_STRONG_MODEL] + list(HEURISTIC_MODELS)
    return {m.safe_name: m for m in models}


def _known_eval_sets() -> Dict[str, EvalSet]:
    return {e.name: e for e in EVAL_SETS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    results_dir = out_dir / "results"
    if not results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {results_dir}")

    summaries: List[dict] = []
    for path in sorted(results_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            summaries.append(json.load(f))
    if not summaries:
        raise SystemExit(f"No result JSON files found under {results_dir}")

    known_models = _known_models()
    models: List[ModelEntry] = []
    seen_model = set()
    for summary in summaries:
        safe = str(summary.get("_model_safe") or "")
        if not safe or safe in seen_model:
            continue
        seen_model.add(safe)
        if safe in known_models:
            models.append(known_models[safe])
        else:
            model = summary.get("model") or {}
            models.append(
                ModelEntry(
                    safe_name=safe,
                    display=str(summary.get("_display") or model.get("name") or safe),
                    kind=str(model.get("kind") or "llm"),
                    model_path=model.get("model_path"),
                    group=str(summary.get("_group") or "trained"),
                )
            )

    known_eval_sets = _known_eval_sets()
    eval_sets: List[EvalSet] = []
    seen_eval = set()
    for summary in summaries:
        name = str(summary.get("_eval_set") or "")
        if not name or name in seen_eval:
            continue
        seen_eval.add(name)
        if name in known_eval_sets:
            eval_sets.append(known_eval_sets[name])
        else:
            eval_sets.append(
                EvalSet(
                    name=name,
                    path="",
                    env=str(summary.get("_env") or ""),
                    flavor=str(summary.get("_flavor") or ""),
                    display=name,
                )
            )

    _write_manifest(out_dir, summaries, models, eval_sets)


if __name__ == "__main__":
    main()
