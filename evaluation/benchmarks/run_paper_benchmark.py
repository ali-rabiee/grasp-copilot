"""
Paper benchmark runner: evaluate every model on every eval set.

Loads each model exactly once and sweeps it across the seven eval sets defined
below (oracle valid × 3 envs, WoZ valid, ambiguous × 3 envs). Per-(model,
eval_set) results are cached as JSON; re-runs skip cells that already exist
unless --rerun is passed.

Outputs land in evaluation/results/paper_benchmark/:
    results/<model_safe>__<eval_set>.json   # one cell each
    mistakes/<model_safe>__<eval_set>.jsonl
    summary_all.csv                          # flat per-cell table
    manifest.json

Usage:
    python -m evaluation.benchmarks.run_paper_benchmark
    python -m evaluation.benchmarks.run_paper_benchmark --models qwen2_5_3b_oracle_woz_lora
    python -m evaluation.benchmarks.run_paper_benchmark --eval_sets oracle_valid_ycb,ambiguous_ycb
    python -m evaluation.benchmarks.run_paper_benchmark --max_examples 50 --skip_zero_shot
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from evaluation import _bootstrap  # noqa: F401
except Exception:
    import _bootstrap  # type: ignore  # noqa: F401

from evaluation.benchmarks.offline_exec_benchmark import (
    ModelSpec,
    _eval_one_model,
    _iter_jsonl,
    _write_csv,
    _write_confusion_matrix_csv,
    _write_context_breakdown_csv,
)


# =============================================================================
# Eval-set & model registries
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class EvalSet:
    name: str            # short stable id used for file names / column headers
    path: str            # contract JSONL (relative to repo root or absolute)
    env: str             # "ycb" | "stacking" | "pouring" | "mixed"
    flavor: str          # "oracle" | "woz" | "ambiguous"
    display: str         # human label for tables


EVAL_SETS: List[EvalSet] = [
    EvalSet("oracle_valid_ycb",      "data/oracle_valid_ycb/llm_contract_200.jsonl",      "ycb",      "oracle",    "Oracle YCB (valid)"),
    EvalSet("oracle_valid_stacking", "data/oracle_valid_stacking/llm_contract_200.jsonl", "stacking", "oracle",    "Oracle Stack (valid)"),
    EvalSet("oracle_valid_pouring",  "data/oracle_valid_pouring/llm_contract_200.jsonl",  "pouring",  "oracle",    "Oracle Pour (valid)"),
    EvalSet("woz_valid",             "data/woz_phase2/llm_contract_valid.jsonl",          "mixed",    "woz",       "WoZ (valid, 3-env)"),
    EvalSet("ambiguous_ycb",         "data/_contracts_ambiguous/ambiguous_reach_to_grasp_ycb.jsonl", "ycb",      "ambiguous", "Ambiguous YCB"),
    EvalSet("ambiguous_stacking",    "data/_contracts_ambiguous/ambiguous_cube_stacking.jsonl",      "stacking", "ambiguous", "Ambiguous Stack"),
    EvalSet("ambiguous_pouring",     "data/_contracts_ambiguous/ambiguous_pouring.jsonl",            "pouring",  "ambiguous", "Ambiguous Pour"),
]


@dataclass(frozen=True)
class ModelEntry:
    safe_name: str          # path-safe id used as file prefix
    display: str            # human label
    kind: str               # "llm" | "heuristic_*"
    model_path: Optional[str] = None
    group: str = "trained"  # "trained" | "ablation" | "baseline" | "zero_shot"


TRAINED_MODELS: List[ModelEntry] = [
    ModelEntry("oracle_lora",       "Qwen2.5-3B-Oracle-LoRA",            "llm", "models/qwen2_5_3b_oracle_lora",            "trained"),
    ModelEntry("woz_lora",          "Qwen2.5-3B-WoZ-LoRA",               "llm", "models/qwen2_5_3b_woz_lora",               "trained"),
    ModelEntry("oracle_woz_lora",   "Qwen2.5-3B-Oracle→WoZ-LoRA",   "llm", "models/qwen2_5_3b_oracle_woz_lora",        "trained"),
    ModelEntry("oracle_woz_r32",    "Qwen2.5-3B-Oracle→WoZ-LoRA-r32","llm", "models/qwen2_5_3b_oracle_woz_lora_r32",   "ablation"),
    ModelEntry("oracle_ycb",        "Qwen2.5-3B-Oracle-LoRA (YCB only)",  "llm", "models/qwen2_5_3b_oracle_lora_ycb",        "ablation"),
    ModelEntry("oracle_stacking",   "Qwen2.5-3B-Oracle-LoRA (Stack only)","llm", "models/qwen2_5_3b_oracle_lora_stacking",  "ablation"),
    ModelEntry("oracle_pouring",    "Qwen2.5-3B-Oracle-LoRA (Pour only)", "llm", "models/qwen2_5_3b_oracle_lora_pouring",   "ablation"),
]

ZERO_SHOT_MODEL = ModelEntry(
    safe_name="qwen3b_zs",
    display="Qwen2.5-3B-Instruct (ZS)",
    kind="llm",
    model_path="Qwen/Qwen2.5-3B-Instruct",
    group="zero_shot",
)

HEURISTIC_MODELS: List[ModelEntry] = [
    ModelEntry("h1_ask_if_amb",     "H1 Ask-if-Ambiguous",   "heuristic_ask_if_ambiguous",   None, "baseline"),
    ModelEntry("h2_always_ask",     "H2 Always-Ask",         "heuristic_always_ask",         None, "baseline"),
    ModelEntry("sa1_pred_assist",   "SA1 Predict-then-Assist","heuristic_predict_then_assist",None, "baseline"),
    ModelEntry("sa2_bayes_intent",  "SA2 Bayesian Intent",   "heuristic_bayesian_intent",    None, "baseline"),
]


# =============================================================================
# Per-cell run
# =============================================================================

def _cell_paths(out_dir: Path, model_safe: str, eval_name: str) -> Tuple[Path, Path]:
    return (
        out_dir / "results"  / f"{model_safe}__{eval_name}.json",
        out_dir / "mistakes" / f"{model_safe}__{eval_name}.jsonl",
    )


def _resolve(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Eval set missing: {path}")
    return list(_iter_jsonl(str(path)))


def _evaluate_cell(
    model: ModelEntry,
    eval_set: EvalSet,
    out_dir: Path,
    *,
    preloaded,
    seed: int,
    max_examples: int,
    use_4bit: bool,
    dump_mistakes: bool,
    rerun: bool,
    progress_every: int,
    max_new_tokens: int = 256,
) -> Dict[str, Any]:
    cell_json, cell_mistakes = _cell_paths(out_dir, model.safe_name, eval_set.name)
    if cell_json.exists() and not rerun:
        with open(cell_json, "r", encoding="utf-8") as f:
            cached = json.load(f)
        cached["_cached"] = True
        return cached

    rows = _load_rows(_resolve(eval_set.path))
    spec = ModelSpec(name=model.display, model_path=model.model_path, kind=model.kind)
    summary = _eval_one_model(
        spec,
        rows,
        seed=seed,
        max_examples=max_examples,
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=int(max_new_tokens),
        use_4bit=use_4bit,
        ignore_interact_text_in_strict=True,
        dump_mistakes_jsonl=cell_mistakes if dump_mistakes else None,
        max_mistakes=200,
        progress_every=progress_every,
        preloaded=preloaded,
    )
    summary["_model_safe"]  = model.safe_name
    summary["_eval_set"]    = eval_set.name
    summary["_env"]         = eval_set.env
    summary["_flavor"]      = eval_set.flavor
    summary["_group"]       = model.group
    summary["_display"]     = model.display

    cell_json.parent.mkdir(parents=True, exist_ok=True)
    with open(cell_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def _flush_model(model_obj, tok_obj) -> None:
    """Aggressively release GPU memory between LLM models."""
    try:
        import torch
        del model_obj, tok_obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        gc.collect()


# =============================================================================
# Orchestration
# =============================================================================

def run(
    *,
    models: List[ModelEntry],
    eval_sets: List[EvalSet],
    out_dir: Path,
    seed: int,
    max_examples: int,
    use_4bit: bool,
    dump_mistakes: bool,
    rerun: bool,
    progress_every: int,
    max_new_tokens: int = 256,
) -> List[Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results").mkdir(exist_ok=True)
    if dump_mistakes:
        (out_dir / "mistakes").mkdir(exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    for mi, model in enumerate(models, start=1):
        print(f"\n{'#'*72}\n# [{mi}/{len(models)}] Model: {model.display} ({model.kind})\n{'#'*72}", flush=True)
        preloaded = None
        if model.kind == "llm":
            from llm.inference import InferenceConfig, _load_model_and_tokenizer
            cfg = InferenceConfig(
                model_path=str(model.model_path),
                use_4bit=bool(use_4bit),
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=int(max_new_tokens),
                seed=int(seed),
                deterministic=True,
            )
            t0 = time.time()
            mdl, tok = _load_model_and_tokenizer(cfg)
            print(f"[{model.display}] loaded in {time.time()-t0:.1f}s", flush=True)
            preloaded = (mdl, tok, cfg)

        for ei, eset in enumerate(eval_sets, start=1):
            print(f"\n  -- [{ei}/{len(eval_sets)}] {eset.name} ({eset.display}) --", flush=True)
            try:
                summary = _evaluate_cell(
                    model, eset, out_dir,
                    preloaded=preloaded,
                    seed=seed,
                    max_examples=max_examples,
                    use_4bit=use_4bit,
                    dump_mistakes=dump_mistakes,
                    rerun=rerun,
                    progress_every=progress_every,
                    max_new_tokens=max_new_tokens,
                )
                summaries.append(summary)
                if summary.get("_cached"):
                    print(f"  [cached] {model.safe_name}__{eset.name}", flush=True)
                else:
                    print(
                        f"  tool_acc={summary['tool_accuracy']:.4f} "
                        f"strict={summary['strict_exact_rate']:.4f} "
                        f"schema={summary['schema_valid_rate']:.4f} "
                        f"n={summary['n']}",
                        flush=True,
                    )
            except Exception as e:
                print(f"  [ERROR] {model.safe_name} on {eset.name}: {e}", flush=True)
                continue

        if preloaded is not None:
            mdl, tok, _ = preloaded
            _flush_model(mdl, tok)

    _write_manifest(out_dir, summaries, models, eval_sets)
    return summaries


# =============================================================================
# Manifest / flat-table writers
# =============================================================================

FLAT_FIELDS = [
    "model_safe", "model_display", "group",
    "eval_set", "env", "flavor",
    "n", "json_valid_rate", "schema_valid_rate",
    "tool_accuracy", "motion_obj_accuracy", "motion_tool_accuracy",
    "interact_kind_accuracy", "interact_choices_valid_rate",
    "strict_exact_rate", "motion_n", "interact_n",
    "examples_per_sec", "eval_s",
]


def _write_manifest(out_dir: Path, summaries: List[Dict[str, Any]], models: List[ModelEntry], eval_sets: List[EvalSet]) -> None:
    manifest = {
        "models": [m.__dict__ for m in models],
        "eval_sets": [e.__dict__ for e in eval_sets],
        "n_cells": len(summaries),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    flat_path = out_dir / "summary_all.csv"
    with open(flat_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FLAT_FIELDS)
        w.writeheader()
        for s in summaries:
            timing = s.get("timing") or {}
            w.writerow({
                "model_safe":     s.get("_model_safe"),
                "model_display":  s.get("_display"),
                "group":          s.get("_group"),
                "eval_set":       s.get("_eval_set"),
                "env":            s.get("_env"),
                "flavor":         s.get("_flavor"),
                "n":              s.get("n"),
                "json_valid_rate":            f"{s.get('json_valid_rate', 0):.4f}",
                "schema_valid_rate":          f"{s.get('schema_valid_rate', 0):.4f}",
                "tool_accuracy":              f"{s.get('tool_accuracy', 0):.4f}",
                "motion_obj_accuracy":        f"{s.get('motion_obj_accuracy', 0):.4f}",
                "motion_tool_accuracy":       f"{s.get('motion_tool_accuracy', 0):.4f}",
                "interact_kind_accuracy":     f"{s.get('interact_kind_accuracy', 0):.4f}",
                "interact_choices_valid_rate":f"{s.get('interact_choices_valid_rate', 0):.4f}",
                "strict_exact_rate":          f"{s.get('strict_exact_rate', 0):.4f}",
                "motion_n":  s.get("motion_n"),
                "interact_n":s.get("interact_n"),
                "examples_per_sec": f"{timing.get('examples_per_sec', 0):.2f}",
                "eval_s":           f"{timing.get('eval_s', 0):.1f}",
            })
    print(f"[manifest] wrote {flat_path} ({len(summaries)} rows)")
    _write_context_breakdown_csv(out_dir / "context_breakdown.csv", summaries)
    _write_confusion_matrix_csv(out_dir / "confusion_matrices.csv", summaries)


# =============================================================================
# CLI
# =============================================================================

def _filter(names: Optional[str], pool, attr: str):
    if not names:
        return list(pool)
    keys = {n.strip() for n in names.split(",") if n.strip()}
    return [p for p in pool if getattr(p, attr) in keys]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default="evaluation/results/paper_benchmark")
    ap.add_argument("--models", default=None, help="Comma-separated safe_names; default = all")
    ap.add_argument("--eval_sets", default=None, help="Comma-separated names; default = all")
    ap.add_argument("--include_zero_shot", action="store_true", help="Also evaluate Qwen2.5-3B-Instruct ZS (downloads from HF)")
    ap.add_argument("--skip_trained", action="store_true")
    ap.add_argument("--skip_heuristics", action="store_true")
    ap.add_argument("--max_examples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--no_dump_mistakes", action="store_true")
    ap.add_argument("--rerun", action="store_true", help="Recompute even if cached cell exists")
    ap.add_argument("--progress_every", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=256,
                    help="Upper bound on generated tokens per call. Tool-call JSONs fit in <128; "
                         "lower for ~2x speedup on chatty LoRAs.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    # Assemble model list in eval order: trained → zero-shot → heuristics.
    selected: List[ModelEntry] = []
    if not args.skip_trained:
        selected.extend(TRAINED_MODELS)
    if args.include_zero_shot:
        selected.append(ZERO_SHOT_MODEL)
    if not args.skip_heuristics:
        selected.extend(HEURISTIC_MODELS)

    if args.models:
        keys = {n.strip() for n in args.models.split(",") if n.strip()}
        selected = [m for m in selected if m.safe_name in keys]

    eval_sets = _filter(args.eval_sets, EVAL_SETS, "name")

    if not selected:
        raise SystemExit("No models selected.")
    if not eval_sets:
        raise SystemExit("No eval sets selected.")

    print(f"[runner] models   = {[m.safe_name for m in selected]}")
    print(f"[runner] eval_sets= {[e.name for e in eval_sets]}")
    print(f"[runner] out_dir  = {out_dir}")

    run(
        models=selected,
        eval_sets=eval_sets,
        out_dir=out_dir,
        seed=int(args.seed),
        max_examples=int(args.max_examples),
        use_4bit=bool(args.use_4bit),
        dump_mistakes=not args.no_dump_mistakes,
        rerun=bool(args.rerun),
        progress_every=int(args.progress_every),
        max_new_tokens=int(args.max_new_tokens),
    )


if __name__ == "__main__":
    main()
