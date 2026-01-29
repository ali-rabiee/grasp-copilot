#!/usr/bin/env python3
"""
Run the full offline executive benchmark for PRIME paper.

This script evaluates:
- Fine-tuned models (Qwen2.5-7B, Qwen2.5-3B)
- Zero-shot models (optional, requires HF access)
- Heuristic baselines (H1: ask-if-ambiguous, H2: always-ask)

Usage:
    # Full benchmark (fine-tuned + heuristics)
    python -m evaluation.run_full_benchmark

    # Include zero-shot baselines (slower, needs HF models)
    python -m evaluation.run_full_benchmark --include_zero_shot

    # Quick test with fewer examples
    python -m evaluation.run_full_benchmark --max_examples 100
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from . import _bootstrap  # noqa: F401
except Exception:
    import _bootstrap  # type: ignore  # noqa: F401


# =============================================================================
# Configuration
# =============================================================================

# Paths relative to grasp-copilot root
DEFAULT_CONTRACT_JSONL = "data/runs/010/llm_contract.jsonl"

# Fine-tuned models
FINETUNED_MODELS = {
    "Qwen2.5-7B-FT": "models/qwen2_5_7b_instruct_ft",
    "Qwen2.5-3B-FT": "models/qwen2_5_3b_instruct_ft",
}

# Zero-shot models (HuggingFace IDs)
ZERO_SHOT_MODELS = {
    "Qwen2.5-7B-ZS": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-3B-ZS": "Qwen/Qwen2.5-3B-Instruct",
    # "Llama-3.1-8B-ZS": "meta-llama/Llama-3.1-8B-Instruct",  # Requires auth
}


def get_grasp_copilot_root() -> Path:
    """Find the grasp-copilot root directory."""
    # Try relative to this file
    this_dir = Path(__file__).resolve().parent
    if (this_dir.parent / "pyproject.toml").exists():
        return this_dir.parent
    # Try current directory
    if (Path.cwd() / "pyproject.toml").exists():
        return Path.cwd()
    # Try parent
    if (Path.cwd().parent / "pyproject.toml").exists():
        return Path.cwd().parent
    raise RuntimeError("Cannot find grasp-copilot root directory")


def discover_models_in_directory(models_dir: Path) -> Dict[str, str]:
    """
    Auto-discover fine-tuned models in the models/ directory.
    
    A valid model directory must contain:
    - config.json (transformers model config)
    OR
    - adapter_config.json (LoRA adapter)
    
    Returns a dict mapping display_name -> model_path (relative to root).
    """
    discovered: Dict[str, str] = {}
    
    if not models_dir.exists() or not models_dir.is_dir():
        return discovered
    
    for subdir in sorted(models_dir.iterdir()):
        if not subdir.is_dir():
            continue
        
        # Check if it's a valid model directory
        has_config = (subdir / "config.json").exists()
        has_adapter = (subdir / "adapter_config.json").exists()
        
        if not (has_config or has_adapter):
            continue
        
        # Generate a display name from directory name
        # e.g., "qwen2_5_1_5b_instruct_ft" -> "Qwen2.5-1.5B-FT"
        dir_name = subdir.name
        
        # Try to infer a nice name
        # Common patterns: qwen2_5_3b_instruct_ft, qwen2_5_7b_instruct_ft, etc.
        display_name = dir_name.replace("_", " ").title().replace(" ", "")
        
        # Try to format it better (e.g., "Qwen25_3B_Instruct_Ft" -> "Qwen2.5-3B-FT")
        # This is a heuristic; users can override via FINETUNED_MODELS if needed
        if "qwen" in dir_name.lower():
            # Extract size (1.5b, 3b, 7b, etc.)
            size_match = re.search(r'(\d+)[._]?(\d*)[bB]', dir_name)
            if size_match:
                major = size_match.group(1)
                minor = size_match.group(2) if size_match.group(2) else ""
                size_str = f"{major}.{minor}B" if minor else f"{major}B"
                display_name = f"Qwen2.5-{size_str}-FT"
            else:
                display_name = f"Qwen-{dir_name}-FT"
        else:
            # Generic fallback
            display_name = f"{dir_name}-FT"
        
        # Use relative path from root
        root = get_grasp_copilot_root()
        try:
            rel_path = subdir.relative_to(root)
            discovered[display_name] = str(rel_path)
        except ValueError:
            # If subdir is not under root, use absolute path
            discovered[display_name] = str(subdir)
    
    return discovered


def run_benchmark(
    contract_jsonl: str,
    models: Dict[str, str],
    out_dir: Path,
    *,
    include_heuristic: bool = True,
    include_heuristic_always_ask: bool = True,
    max_examples: int = 0,
    seed: int = 0,
    use_4bit: bool = False,
    dump_mistakes: bool = True,
    progress_every: int = 100,
) -> Dict[str, Any]:
    """Run the benchmark using offline_exec_benchmark module."""
    from evaluation.offline_exec_benchmark import (
        ModelSpec,
        _eval_one_model,
        _iter_jsonl,
        _write_json,
        _write_csv,
        _write_context_breakdown_csv,
        _write_confusion_matrix_csv,
    )

    rows = list(_iter_jsonl(contract_jsonl))
    if not rows:
        raise RuntimeError(f"Empty contract JSONL: {contract_jsonl}")
    print(f"[benchmark] Loaded {len(rows)} examples from {contract_jsonl}")

    # Build model specs
    specs: List[ModelSpec] = []
    for name, path in models.items():
        specs.append(ModelSpec(name=name, model_path=path, kind="llm"))
    if include_heuristic:
        specs.append(ModelSpec(name="H1_ask_if_ambiguous", kind="heuristic_ask_if_ambiguous"))
    if include_heuristic_always_ask:
        specs.append(ModelSpec(name="H2_always_ask", kind="heuristic_always_ask"))

    print(f"[benchmark] Evaluating {len(specs)} model(s): {[s.name for s in specs]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    all_summaries: List[Dict[str, Any]] = []

    for spec in specs:
        print(f"\n{'='*60}\n[benchmark] Evaluating: {spec.name}\n{'='*60}")
        mistakes_path = None
        if dump_mistakes:
            safe = spec.name.replace("/", "_").replace(" ", "_")
            mistakes_path = out_dir / f"mistakes_{safe}.jsonl"
        
        summary = _eval_one_model(
            spec,
            rows,
            seed=seed,
            max_examples=max_examples,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=256,
            use_4bit=use_4bit,
            ignore_interact_text_in_strict=True,
            dump_mistakes_jsonl=mistakes_path,
            max_mistakes=200,
            progress_every=progress_every,
        )
        all_summaries.append(summary)

        # Print summary
        print(f"\n[{spec.name}] Results:")
        print(f"  Tool accuracy:        {summary['tool_accuracy']:.4f}")
        print(f"  Motion obj accuracy:  {summary['motion_obj_accuracy']:.4f}")
        print(f"  Interact kind acc:    {summary['interact_kind_accuracy']:.4f}")
        print(f"  Schema valid rate:    {summary['schema_valid_rate']:.4f}")

    # Write outputs
    _write_json(out_dir / "summary_all.json", {"contract_jsonl": contract_jsonl, "summaries": all_summaries})
    _write_csv(out_dir / "summary_all.csv", all_summaries)
    _write_context_breakdown_csv(out_dir / "context_breakdown.csv", all_summaries)
    _write_confusion_matrix_csv(out_dir / "confusion_matrices.csv", all_summaries)

    return {"summaries": all_summaries, "out_dir": str(out_dir)}


def print_results_table(summaries: List[Dict[str, Any]]) -> None:
    """Print a formatted results table."""
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 100)
    
    # Header
    header = f"{'Model':<25} {'Tool Acc':>10} {'Mot.Obj':>10} {'Int.Kind':>10} {'Schema':>10} {'Strict':>10} {'N':>8}"
    print(header)
    print("-" * 100)
    
    for s in summaries:
        name = s.get("model", {}).get("name", "unknown")[:24]
        tool_acc = s.get("tool_accuracy", 0)
        mot_obj = s.get("motion_obj_accuracy", 0)
        int_kind = s.get("interact_kind_accuracy", 0)
        schema = s.get("schema_valid_rate", 0)
        strict = s.get("strict_exact_rate", 0)
        n = s.get("n", 0)
        
        row = f"{name:<25} {tool_acc:>10.4f} {mot_obj:>10.4f} {int_kind:>10.4f} {schema:>10.4f} {strict:>10.4f} {n:>8}"
        print(row)
    
    print("=" * 100)


def print_context_breakdown(summaries: List[Dict[str, Any]]) -> None:
    """Print per-context accuracy breakdown."""
    print("\n" + "=" * 100)
    print("PER-CONTEXT TOOL ACCURACY BREAKDOWN")
    print("=" * 100)
    
    # Collect all contexts
    all_contexts = set()
    for s in summaries:
        all_contexts.update(s.get("by_context", {}).keys())
    contexts = sorted(all_contexts)
    
    # Header
    header = f"{'Model':<20}"
    for ctx in contexts[:8]:  # Limit to 8 contexts for display
        header += f" {ctx[:12]:>12}"
    print(header)
    print("-" * 100)
    
    for s in summaries:
        name = s.get("model", {}).get("name", "unknown")[:19]
        row = f"{name:<20}"
        by_ctx = s.get("by_context", {})
        for ctx in contexts[:8]:
            ctx_data = by_ctx.get(ctx, {})
            n = ctx_data.get("n", 0)
            correct = ctx_data.get("tool_correct", 0)
            acc = correct / n if n > 0 else 0
            row += f" {acc:>12.3f}"
        print(row)
    
    print("=" * 100)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run full offline executive benchmark for PRIME paper.")
    ap.add_argument("--contract_jsonl", type=str, default=None, help="Path to contract JSONL (default: data/runs/010/llm_contract.jsonl)")
    ap.add_argument("--out_dir", type=str, default=None, help="Output directory (default: auto-generated timestamp)")
    ap.add_argument("--include_zero_shot", action="store_true", help="Include zero-shot model baselines (slower)")
    ap.add_argument("--skip_finetuned", action="store_true", help="Skip fine-tuned models")
    ap.add_argument("--skip_heuristics", action="store_true", help="Skip heuristic baselines")
    ap.add_argument("--max_examples", type=int, default=0, help="Max examples (0=all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")
    ap.add_argument("--no_dump_mistakes", action="store_true", help="Don't dump mistake JSONLs")
    ap.add_argument("--progress_every", type=int, default=100)
    args = ap.parse_args()

    root = get_grasp_copilot_root()
    print(f"[benchmark] grasp-copilot root: {root}")

    # Determine contract JSONL path
    if args.contract_jsonl:
        contract_jsonl = args.contract_jsonl
    else:
        contract_jsonl = str(root / DEFAULT_CONTRACT_JSONL)
    
    if not Path(contract_jsonl).exists():
        raise SystemExit(f"Contract JSONL not found: {contract_jsonl}")

    # Determine output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = root / "evaluation" / "eval_outputs" / f"benchmark_{timestamp}"

    # Build model list
    models: Dict[str, str] = {}
    
    if not args.skip_finetuned:
        # First, add hardcoded models (these take precedence)
        for name, rel_path in FINETUNED_MODELS.items():
            full_path = root / rel_path
            if full_path.exists():
                models[name] = str(full_path)
                print(f"[benchmark] Found fine-tuned model (hardcoded): {name} -> {full_path}")
            else:
                print(f"[benchmark] WARNING: Fine-tuned model not found: {full_path}")
        
        # Then, auto-discover additional models in models/ directory
        models_dir = root / "models"
        discovered = discover_models_in_directory(models_dir)
        for name, rel_path in discovered.items():
            # Skip if already in models (hardcoded takes precedence)
            if name not in models:
                full_path = root / rel_path
                if full_path.exists():
                    models[name] = str(full_path)
                    print(f"[benchmark] Found fine-tuned model (auto-discovered): {name} -> {full_path}")
    
    if args.include_zero_shot:
        for name, hf_id in ZERO_SHOT_MODELS.items():
            models[name] = hf_id
            print(f"[benchmark] Will evaluate zero-shot: {name} -> {hf_id}")

    if not models and args.skip_heuristics:
        raise SystemExit("No models to evaluate (all skipped)")

    # Run benchmark
    result = run_benchmark(
        contract_jsonl=contract_jsonl,
        models=models,
        out_dir=out_dir,
        include_heuristic=not args.skip_heuristics,
        include_heuristic_always_ask=not args.skip_heuristics,
        max_examples=args.max_examples,
        seed=args.seed,
        use_4bit=args.use_4bit,
        dump_mistakes=not args.no_dump_mistakes,
        progress_every=args.progress_every,
    )

    # Print formatted results
    print_results_table(result["summaries"])
    print_context_breakdown(result["summaries"])

    print(f"\n[benchmark] All outputs written to: {out_dir}")
    print(f"  - summary_all.json / summary_all.csv")
    print(f"  - context_breakdown.csv")
    print(f"  - confusion_matrices.csv")
    if not args.no_dump_mistakes:
        print(f"  - mistakes_*.jsonl")


if __name__ == "__main__":
    main()



