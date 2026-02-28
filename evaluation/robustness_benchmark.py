"""
Robustness-to-perturbation benchmark for PRIME.

Evaluates how heuristic baselines (rule-based, oracle-like) and
fine-tuned LLM models degrade when the symbolic state is noisy.

Usage:
    # Heuristics only (instant, no GPU):
    python evaluation/robustness_benchmark.py \
        --contract_jsonl data/runs/001/llm_contract.jsonl \
        --perturbations user_input \
        --max_examples 0

    # With fine-tuned model:
    python evaluation/robustness_benchmark.py \
        --contract_jsonl data/runs/001/llm_contract.jsonl \
        --model_path models/qwen2_5_7b_instruct_ft \
        --model_name ft_7b \
        --perturbations user_input \
        --max_examples 0
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    from . import _bootstrap  # noqa: F401
except Exception:
    import _bootstrap  # type: ignore  # noqa: F401

from data_generator.grid import CELLS, manhattan as grid_manhattan, neighbors as grid_neighbors
from data_generator.episode import OBJECT_LABELS
from data_generator.yaw import YAW_BINS, neighbors as yaw_neighbors

CANDIDATE_MAX_DIST = 1

from llm.utils import json_loads_strict, set_seed

from evaluation.offline_exec_benchmark import (
    _heuristic_ask_if_ambiguous,
    _heuristic_always_ask,
    _normalize_tool_call,
    _parse_model_json,
    _rate,
    _iter_jsonl,
)


# ── perturbation functions ──────────────────────────────────────────────

def _jitter_cell(cell: str, rng: random.Random) -> str:
    nbrs = grid_neighbors(cell)
    return rng.choice(nbrs) if nbrs else cell


def _recompute_candidates(out: Dict) -> None:
    """Recalculate candidates from perturbed positions (mirrors episode.gripper_candidates)."""
    gripper_cell = out.get("gripper_hist", [{}])[-1].get("cell", "B2")
    new_cands = []
    for obj in out.get("objects", []):
        if obj.get("is_held"):
            continue
        if grid_manhattan(gripper_cell, obj["cell"]) <= CANDIDATE_MAX_DIST:
            new_cands.append(obj["id"])
    out.setdefault("memory", {})["candidates"] = new_cands


def perturb_grid_jitter(inp: Dict, p: float, rng: random.Random) -> Dict:
    out = copy.deepcopy(inp)
    for obj in out.get("objects", []):
        if rng.random() < p:
            obj["cell"] = _jitter_cell(obj["cell"], rng)
    for pose in out.get("gripper_hist", []):
        if rng.random() < p:
            pose["cell"] = _jitter_cell(pose["cell"], rng)
    _recompute_candidates(out)
    return out


def perturb_candidate_set(inp: Dict, p: float, rng: random.Random) -> Dict:
    out = copy.deepcopy(inp)
    mem = out.get("memory", {})
    candidates = list(mem.get("candidates", []))
    all_ids = [o["id"] for o in out.get("objects", [])]
    cand_set = set(candidates)

    kept = [c for c in candidates if rng.random() >= p]
    non_cands = [oid for oid in all_ids if oid not in cand_set]
    added = [nc for nc in non_cands if rng.random() < p]

    mem["candidates"] = kept + added
    return out


def perturb_user_input(inp: Dict, p: float, rng: random.Random) -> Dict:
    """Simulate noisy user teleoperation: jitter gripper-history cells and yaw bins.

    This is the most deployment-realistic perturbation: object positions come
    from a calibrated camera and are stable, but the user's control signal
    (joystick, head-array, sip-and-puff, etc.) is inherently noisy.  The oracle
    relies on exact cell counts, yaw-switch counts, and oscillation thresholds
    over the 6-pose gripper history — all of which are brittle to even single-
    step input noise.
    """
    out = copy.deepcopy(inp)
    for pose in out.get("gripper_hist", []):
        if rng.random() < p:
            pose["cell"] = _jitter_cell(pose["cell"], rng)
        if rng.random() < p:
            nbrs = yaw_neighbors(pose["yaw"])
            pose["yaw"] = rng.choice(nbrs)
    _recompute_candidates(out)
    return out


def perturb_label_noise(inp: Dict, p: float, rng: random.Random) -> Dict:
    out = copy.deepcopy(inp)
    for obj in out.get("objects", []):
        if rng.random() < p:
            alts = [l for l in OBJECT_LABELS if l != obj["label"]]
            obj["label"] = rng.choice(alts)
    return out


PERTURBATION_REGISTRY: Dict[str, Callable] = {
    "user_input": perturb_user_input,
    "grid_jitter": perturb_grid_jitter,
    "candidate_perturb": perturb_candidate_set,
    "label_noise": perturb_label_noise,
}


# ── scoring ─────────────────────────────────────────────────────────────

def _score(gt: Dict, pred: Dict, counters: Dict[str, int]) -> None:
    counters["n"] += 1
    gt_tool = str(gt.get("tool"))
    pr_tool = str(pred.get("tool"))

    if gt_tool == pr_tool:
        counters["tool_correct"] += 1

    gt2, pr2 = copy.deepcopy(gt), copy.deepcopy(pred)
    if gt2.get("tool") == "INTERACT" and pr2.get("tool") == "INTERACT":
        if isinstance(gt2.get("args"), dict):
            gt2["args"].pop("text", None)
        if isinstance(pr2.get("args"), dict):
            pr2["args"].pop("text", None)
    if gt2 == pr2:
        counters["strict_exact"] += 1

    if gt_tool in {"APPROACH", "ALIGN_YAW"}:
        counters["motion_n"] += 1
        if pr_tool in {"APPROACH", "ALIGN_YAW"}:
            gt_obj = (gt.get("args") or {}).get("obj")
            pr_obj = (pred.get("args") or {}).get("obj")
            if gt_obj == pr_obj:
                counters["motion_obj_correct"] += 1


def _fresh_counters() -> Dict[str, int]:
    return {"n": 0, "tool_correct": 0, "strict_exact": 0, "motion_n": 0, "motion_obj_correct": 0}


# ── main ────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Robustness-to-perturbation benchmark.")
    ap.add_argument("--contract_jsonl", required=True)
    ap.add_argument("--model_path", default=None, help="Path to fine-tuned model (omit for heuristics-only run).")
    ap.add_argument("--model_name", default="ft_7b")
    ap.add_argument("--perturbations", nargs="+", default=["user_input"],
                     choices=list(PERTURBATION_REGISTRY))
    ap.add_argument("--noise_levels", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--max_examples", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--out_dir", default="evaluation/eval_outputs/robustness")
    ap.add_argument("--progress_every", type=int, default=50)
    args = ap.parse_args(argv)

    set_seed(args.seed)
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(_iter_jsonl(args.contract_jsonl))
    if not rows:
        raise SystemExit("Empty contract JSONL")
    n = min(args.max_examples, len(rows)) if args.max_examples > 0 else len(rows)
    sample = rows[:n]
    print(f"[robustness] {len(sample)} examples, perturbations={args.perturbations}, "
          f"noise_levels={args.noise_levels}")

    # optionally load LLM
    model = tok = cfg = None
    if args.model_path:
        from llm.inference import InferenceConfig, _build_messages, _generate_once, _load_model_and_tokenizer
        cfg = InferenceConfig(
            model_path=args.model_path,
            use_4bit=args.use_4bit,
            temperature=0.0, top_p=1.0,
            max_new_tokens=256,
            seed=args.seed, deterministic=True,
        )
        t0 = time.time()
        model, tok = _load_model_and_tokenizer(cfg)
        print(f"[robustness] Model loaded in {time.time()-t0:.1f}s")

    all_results: List[Dict[str, Any]] = []

    for perturb_name in args.perturbations:
        perturb_fn = PERTURBATION_REGISTRY[perturb_name]

        for p in args.noise_levels:
            rng_run = random.Random(args.seed)

            systems: Dict[str, Dict[str, int]] = {
                "H1": _fresh_counters(),
                "H2": _fresh_counters(),
            }
            if model:
                systems[args.model_name] = _fresh_counters()

            t0 = time.time()
            for idx, row in enumerate(sample):
                instruction = str(row.get("instruction", ""))
                input_str = str(row.get("input", ""))
                gt_str = str(row.get("output", ""))

                try:
                    gt = _normalize_tool_call(json_loads_strict(gt_str))
                    inp = json_loads_strict(input_str)
                except Exception:
                    continue

                perturbed = perturb_fn(inp, p, rng_run)
                p_str = json.dumps(perturbed, ensure_ascii=False)

                # H1
                try:
                    h1 = _normalize_tool_call(_heuristic_ask_if_ambiguous(p_str))
                    _score(gt, h1, systems["H1"])
                except Exception:
                    systems["H1"]["n"] += 1

                # H2
                try:
                    h2 = _normalize_tool_call(_heuristic_always_ask(p_str))
                    _score(gt, h2, systems["H2"])
                except Exception:
                    systems["H2"]["n"] += 1

                # FT model
                if model and tok and cfg:
                    from llm.inference import _build_messages, _generate_once
                    prompt = f"{instruction}\n\nInput:\n{p_str}"
                    try:
                        raw = _generate_once(model, tok, _build_messages(prompt), cfg)
                        pred_obj, _ = _parse_model_json(raw)
                        if pred_obj:
                            _score(gt, _normalize_tool_call(pred_obj), systems[args.model_name])
                        else:
                            systems[args.model_name]["n"] += 1
                    except Exception:
                        systems[args.model_name]["n"] += 1

                if args.progress_every > 0 and (idx + 1) % args.progress_every == 0:
                    elapsed = time.time() - t0
                    rate = (idx + 1) / max(elapsed, 1e-6)
                    print(f"  [{perturb_name} p={p}] {idx+1}/{len(sample)} | {rate:.1f} ex/s")

            elapsed = time.time() - t0
            for sys_name, c in systems.items():
                rec = {
                    "perturbation": perturb_name,
                    "noise_level": p,
                    "system": sys_name,
                    "n": c["n"],
                    "tool_acc": round(_rate(c["tool_correct"], c["n"]) * 100, 2),
                    "strict_match": round(_rate(c["strict_exact"], c["n"]) * 100, 2),
                    "motion_obj_acc": round(_rate(c["motion_obj_correct"], c["motion_n"]) * 100, 2),
                    "elapsed_s": round(elapsed, 1),
                }
                all_results.append(rec)
                print(f"  {sys_name:12s} | tool_acc={rec['tool_acc']:6.2f}%  strict={rec['strict_match']:6.2f}%  "
                      f"mot_obj={rec['motion_obj_acc']:6.2f}%")

    # write CSV
    csv_path = out_dir / "robustness_results.csv"
    fieldnames = ["perturbation", "noise_level", "system", "n", "tool_acc", "strict_match", "motion_obj_acc", "elapsed_s"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_results:
            w.writerow(r)

    # write JSON
    json_path = out_dir / "robustness_results.json"
    with open(json_path, "w") as f:
        json.dump({"config": vars(args), "results": all_results}, f, indent=2)

    print(f"\n[robustness] Results saved to {csv_path}")
    print(f"[robustness] Full config+results saved to {json_path}")


if __name__ == "__main__":
    main()
