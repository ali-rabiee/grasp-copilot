"""
Scenario-seeded noise-robustness sweep (plan §5.7).

Runs the (scenario × noise condition × mode × seed) matrix and streams
per-rollout records to CSV. Then aggregates into a by-condition summary.

Default matrix size: 160 scenarios × 6 conditions × 2 modes × 10 seeds
= **19,200 rollouts** per backend.

Backends
--------
  * ``manual``    — no LLM. Scripted user drives end-to-end. Used for the
                    Manual baseline; ignored in PRIME mode (PRIME requires
                    a non-manual backend).
  * ``defer``     — PRIME mode that always defers to the user. Equivalent to
                    manual but routes through the PRIME loop, useful as a
                    sanity check that PRIME mode is mechanically correct.
  * ``oracle``    — heuristic oracle as a stateful PRIME stand-in. Each
                    rollout gets its own ``OracleBackend`` instance which
                    maintains ``OracleState`` and applies state updates from
                    the user's reply (a stripped-down port of
                    ``data_generator.generate_dataset._simulate_user_response``).
                    Cheap, no GPU.
  * ``hf_ft``     — fine-tuned Qwen LLM via ``llm.inference``. Stateless
                    (re-reads ``memory.past_dialogs`` each call). Single-
                    process; model cached at module level. Use ``--model_paths
                    name=path`` to specify one or more checkpoints.
  * ``heuristic`` — one of h1/h2/sa1/sa2 from
                    ``offline_exec_benchmark`` lifted into the rollout loop.
                    Stateless, CPU-only. Use ``--heuristic <name>``.

CLI
---
    # Manual baseline (CPU, multiprocess):
    python -m evaluation.benchmarks.scenario_noise_sweep \\
        --scenarios evaluation/results/robustness/user_input_noise/scenarios/scenarios.labeled.jsonl \\
        --out_dir   evaluation/results/robustness/user_input_noise/sweeps/manual \\
        --modes manual --backend manual --n_seeds 5 --workers 4

    # Headline + warm-start ablation (GPU, single-process per model):
    python -m evaluation.benchmarks.scenario_noise_sweep \\
        --scenarios evaluation/results/robustness/user_input_noise/scenarios/scenarios.labeled.jsonl \\
        --out_dir   evaluation/results/robustness/user_input_noise/sweeps/llm \\
        --modes prime --backend hf_ft \\
        --model_paths oracle_woz_lora=models/qwen2_5_3b_oracle_woz_lora \\
                      oracle_lora=models/qwen2_5_3b_oracle_lora \\
        --n_seeds 5

    # Heuristic baseline (CPU, multiprocess):
    python -m evaluation.benchmarks.scenario_noise_sweep \\
        --scenarios evaluation/results/robustness/user_input_noise/scenarios/scenarios.labeled.jsonl \\
        --out_dir   evaluation/results/robustness/user_input_noise/sweeps/heuristic \\
        --modes prime --backend heuristic --heuristic h1_ask_if_amb \\
        --n_seeds 5 --workers 4

Output
------
    out_dir/rollouts.csv         per-rollout records, one row per
                                  (scenario × mode × condition × seed)
    out_dir/by_condition.csv     aggregated means / success rates
    out_dir/sweep_meta.json      run config + timing
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from evaluation import _bootstrap  # noqa: F401
except Exception:
    import _bootstrap  # type: ignore  # noqa: F401

from evaluation.scenarios import load_scenarios
from evaluation.rollouts.noise import STANDARD_CONDITIONS, NoiseProfile
from evaluation.rollouts.rollout_loop import run_rollout


CONDITION_LOOKUP = {p.name: p for p in STANDARD_CONDITIONS}

ROLLOUT_FIELDS = [
    "scenario_id", "subject", "difficulty", "trial_mode",
    "mode", "condition", "seed",
    "success", "completion_time_sec",
    "total_inputs", "interactions", "motion_tool_calls",
    "mode_switches", "direction_reversals",
    "dropped_inputs", "selection_perturbations", "direction_perturbations",
    "target_filtered_out", "terminated_at_max_ticks", "end_reason",
]


# ── oracle backend (stateful, reply-aware) ─────────────────────────────────


class OracleBackend:
    """Heuristic oracle as a stateful PRIME backend.

    The rollout loop calls ``__call__(input_dict)`` for each PRIME decision
    and ``on_user_reply(...)`` after the scripted user answers an INTERACT.
    The reply handler is a stripped-down port of
    ``data_generator.generate_dataset._simulate_user_response`` — it covers
    the context types the oracle actually emits for the reach_to_grasp_ycb
    env (confirm / candidate_choice / anything_else / mode_select / help /
    intent_gate_*).
    """

    def __init__(self, target_obj_id: str):
        from data_generator.oracle import OracleState
        self._oracle_state_cls = OracleState
        self.state = OracleState(intended_obj_id=target_obj_id)
        self.target_obj_id = target_obj_id

    def __call__(self, input_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self.state.terminate_episode:
            return None
        from data_generator.oracle import oracle_decide_tool
        try:
            return oracle_decide_tool(
                input_dict["objects"], input_dict["gripper_hist"],
                input_dict["memory"], self.state,
                user_state=input_dict["user_state"],
            )
        except Exception:
            return None

    def on_user_reply(self, tool_call: Dict[str, Any], reply_idx: int, memory: Dict[str, Any], episode) -> None:
        if tool_call.get("tool") != "INTERACT":
            self.state.last_prompt_context = None
            return

        ctx = self.state.last_prompt_context or {}
        ctx_type = ctx.get("type", "")
        choices: List[str] = list(tool_call.get("args", {}).get("choices") or [])
        choice_str = choices[reply_idx] if 0 <= reply_idx < len(choices) else ""
        upper = choice_str.upper()
        is_yes = ("YES" in upper) or ("OK" in upper) or ("CONFIRM" in upper)
        is_none = ("NONE OF THEM" in upper) or (choice_str.strip().lower() == "none")

        if ctx_type in (
            "confirm", "help", "confirm_stack", "confirm_pour", "confirm_grab",
            "pitcher_acquisition", "non_top_redirect", "cup_full_redirect",
        ):
            obj_id = ctx.get("obj_id") or ctx.get("alt_obj_id")
            if is_yes and obj_id:
                self.state.pending_action_obj_id = obj_id
                self.state.selected_obj_id = obj_id
                self.state.intended_obj_id = obj_id
                action = str(ctx.get("action") or "").upper()
                if action in {"APPROACH", "ALIGN_YAW", "STACK", "GRAB", "POUR"}:
                    self.state.pending_mode = action
                self.state.last_declined_obj_id = None
            else:
                if obj_id:
                    self.state.last_declined_obj_id = obj_id
                    ex = set(memory.get("excluded_obj_ids") or [])
                    ex.add(str(obj_id))
                    memory["excluded_obj_ids"] = sorted(ex)
                self.state.pending_action_obj_id = None
                self.state.pending_mode = None
                self.state.awaiting_anything_else = True
            if ctx_type in {"confirm", "help", "confirm_stack", "confirm_pour", "confirm_grab"}:
                self.state.selected_obj_id = None
            self.state.awaiting_confirmation = False
            self.state.awaiting_help = False

        elif ctx_type == "candidate_choice":
            obj_ids: List[str] = list(ctx.get("obj_ids") or [])
            none_index_1 = int(ctx.get("none_index") or (len(choices) + 1))
            if is_none or (reply_idx == none_index_1 - 1):
                # User said "None of them" — exclude all the listed objects.
                ex = set(memory.get("excluded_obj_ids") or [])
                for oid in obj_ids:
                    ex.add(oid)
                memory["excluded_obj_ids"] = sorted(ex)
                self.state.awaiting_choice = True
                self.state.awaiting_confirmation = False
                self.state.selected_obj_id = None
                self.state.last_prompt_context = None
                return
            if 0 <= reply_idx < len(obj_ids):
                self.state.selected_obj_id = obj_ids[reply_idx]
                self.state.intended_obj_id = obj_ids[reply_idx]
            self.state.awaiting_choice = False
            self.state.awaiting_confirmation = False

        elif ctx_type == "anything_else":
            if is_yes:
                self.state.awaiting_mode_select = True
                self.state.awaiting_anything_else = False
            else:
                self.state.terminate_episode = True
                self.state.awaiting_anything_else = False

        elif ctx_type == "mode_select":
            actions: List[str] = list(ctx.get("actions") or ["APPROACH", "ALIGN_YAW"])
            pick = next((a for a in actions if a in upper), actions[0])
            self.state.pending_mode = pick
            self.state.awaiting_mode_select = False
            self.state.awaiting_choice = True

        elif ctx_type in {
            "intent_gate_candidates", "intent_gate_yaw",
            "intent_gate_stack", "intent_gate_pour",
        }:
            if is_yes:
                if ctx_type == "intent_gate_yaw":
                    self.state.awaiting_help = True
                    self.state.awaiting_intent_gate = False
                    self.state.pending_mode = "ALIGN_YAW"
                    inferred = ctx.get("obj_id")
                    if inferred:
                        self.state.selected_obj_id = inferred
                else:
                    self.state.awaiting_choice = True
                    self.state.awaiting_intent_gate = False
                    action = str(ctx.get("action") or "APPROACH").upper()
                    self.state.pending_mode = action
            else:
                self.state.awaiting_intent_gate = False
                self.state.awaiting_choice = False
                self.state.awaiting_help = False
                self.state.awaiting_confirmation = False
                self.state.awaiting_anything_else = True

        self.state.last_prompt_context = None


HEURISTIC_REGISTRY: Dict[str, str] = {
    "h1_ask_if_amb":     "_heuristic_ask_if_ambiguous",
    "h2_always_ask":     "_heuristic_always_ask",
    "sa1_pred_assist":   "_heuristic_predict_then_assist",
    "sa2_bayes_intent":  "_heuristic_bayesian_intent",
}


def _build_backend(backend_name: str, target_obj_id: str, model_path: Optional[str] = None,
                    heuristic_name: Optional[str] = None):
    if backend_name == "manual" or backend_name == "defer":
        return None
    if backend_name == "oracle":
        return OracleBackend(target_obj_id=target_obj_id)
    if backend_name == "hf_ft":
        return _build_hf_backend(model_path)
    if backend_name == "heuristic":
        return _build_heuristic_backend(heuristic_name)
    raise ValueError(f"unknown backend {backend_name!r}")


def _build_heuristic_backend(heuristic_name: Optional[str]):
    """Stateless heuristic backend for PRIME mode.

    Wraps the four heuristic / shared-autonomy decision functions already in
    ``evaluation/benchmarks/offline_exec_benchmark.py`` (h1/h2/sa1/sa2). The
    heuristic functions take a JSON-serialized input string and return a
    tool-call dict; this closure just adapts the calling convention. No
    state, no GPU.
    """
    if not heuristic_name or heuristic_name not in HEURISTIC_REGISTRY:
        raise ValueError(
            f"--backend heuristic requires --heuristic one of {list(HEURISTIC_REGISTRY)} "
            f"(got {heuristic_name!r})"
        )

    from evaluation.benchmarks import offline_exec_benchmark as _ob
    fn_name = HEURISTIC_REGISTRY[heuristic_name]
    decide_fn = getattr(_ob, fn_name)

    def backend(input_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            return decide_fn(json.dumps(input_dict, ensure_ascii=False))
        except Exception:
            return None

    return backend


# Module-level singleton for the loaded HF model so we don't pay the load
# cost per rollout. Only one model is held at a time (we clear between
# sweeps when iterating over multiple models). Workers in multiprocessing
# DO NOT share this cache — hf_ft is single-process by design.
_HF_CACHE: Dict[str, Any] = {}


def _build_hf_backend(model_path: Optional[str]):
    """Fine-tuned Qwen LLM backend for PRIME mode.

    The LLM is stateless: it re-reads ``memory.past_dialogs`` and the rest of
    the input each call, so this backend has no ``on_user_reply`` method.
    The model is loaded once and cached at module level; the returned closure
    just builds the prompt, calls ``_generate_once``, and parses the JSON.

    Deterministic (temperature=0, max_new_tokens=256) so rollouts are
    reproducible at fixed (seed, model_path).
    """
    if not model_path:
        raise ValueError("hf_ft backend requires --model_paths (one or more name=path)")

    if "model" not in _HF_CACHE or _HF_CACHE.get("path") != model_path:
        from llm.inference import (
            InferenceConfig,
            _build_messages,
            _generate_once,
            _load_model_and_tokenizer,
        )
        cfg = InferenceConfig(
            model_path=model_path,
            use_4bit=False,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=256,
            seed=0,
            deterministic=True,
        )
        print(f"[sweep] loading HF model from {model_path} (one-time cost)...")
        t0 = time.time()
        model, tok = _load_model_and_tokenizer(cfg)
        print(f"[sweep] model loaded in {time.time() - t0:.1f}s")
        _HF_CACHE.clear()
        _HF_CACHE["model"], _HF_CACHE["tok"], _HF_CACHE["cfg"] = model, tok, cfg
        _HF_CACHE["path"] = model_path
        _HF_CACHE["build_messages"] = _build_messages
        _HF_CACHE["generate_once"] = _generate_once

    model = _HF_CACHE["model"]
    tok = _HF_CACHE["tok"]
    cfg = _HF_CACHE["cfg"]
    build_messages = _HF_CACHE["build_messages"]
    generate_once = _HF_CACHE["generate_once"]

    from evaluation.benchmarks.offline_exec_benchmark import _parse_model_json

    instruction = (
        "Given the robot observation and dialog context, infer the user's intent and "
        "emit exactly one tool call. Output ONLY the tool call JSON with keys tool and args. "
        "If the tool is INTERACT, you must output at most 5 choices total."
    )

    def backend(input_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        prompt = f"{instruction}\n\nInput:\n{json.dumps(input_dict, ensure_ascii=False)}"
        try:
            raw = generate_once(model, tok, build_messages(prompt), cfg)
            pred_obj, _ = _parse_model_json(raw)
            return pred_obj
        except Exception:
            return None

    return backend


def _free_hf_cache() -> None:
    """Drop the cached HF model so the next model load can claim its VRAM."""
    if not _HF_CACHE:
        return
    _HF_CACHE.clear()
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ── single-rollout worker ──────────────────────────────────────────────────


def _run_one(job: Dict[str, Any]) -> Dict[str, Any]:
    """Run one rollout and return its CSV row dict."""
    scenarios = load_scenarios(job["scenarios_path"])
    by_id = {s.scenario_id: s for s in scenarios}
    s = by_id[job["scenario_id"]]

    profile = CONDITION_LOOKUP[job["condition"]]
    backend_name = "manual" if job["mode"] == "manual" else job["backend"]
    backend = _build_backend(
        backend_name,
        target_obj_id=s.target_obj_id,
        model_path=job.get("model_path"),
        heuristic_name=job.get("heuristic_name"),
    )

    r = run_rollout(
        s, mode=job["mode"], noise_profile=profile, seed=job["seed"],
        backend=backend, max_ticks=job["max_ticks"],
        use_per_scenario_priors=job["use_per_scenario_priors"],
    )

    row = {
        "scenario_id":   s.scenario_id,
        "subject":       s.source.subject,
        "difficulty":    s.source.difficulty,
        "trial_mode":    s.source.mode,
        "mode":          r.mode,
        "condition":     r.condition,
        "seed":          r.seed,
        "success":       int(r.success),
        "completion_time_sec":        round(r.completion_time_sec, 4),
        "total_inputs":               r.total_inputs,
        "interactions":               r.interactions,
        "motion_tool_calls":          r.motion_tool_calls,
        "mode_switches":              r.mode_switches,
        "direction_reversals":        r.direction_reversals,
        "dropped_inputs":             r.dropped_inputs,
        "selection_perturbations":    r.selection_perturbations,
        "direction_perturbations":    r.direction_perturbations,
        "target_filtered_out":        int(r.target_filtered_out),
        "terminated_at_max_ticks":    int(r.terminated_at_max_ticks),
        "end_reason":                 r.end_reason,
    }
    return row


# ── aggregation ────────────────────────────────────────────────────────────


def aggregate(rollouts_csv: Path, out_csv: Path) -> None:
    """Aggregate per-rollout records into a by-(mode, condition, difficulty) summary."""
    from collections import defaultdict
    import statistics as st

    rows = list(csv.DictReader(rollouts_csv.open()))
    buckets: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[(r["mode"], r["condition"], r["difficulty"])].append(r)

    def _safe_mean(vals): return st.mean(vals) if vals else 0.0
    def _safe_std(vals):  return st.stdev(vals) if len(vals) > 1 else 0.0

    out_fields = [
        "mode", "condition", "difficulty",
        "n_rollouts", "success_rate",
        "mean_completion_time_sec", "std_completion_time_sec",
        "mean_total_inputs", "std_total_inputs",
        "mean_interactions", "mean_dropped_inputs",
        "mean_direction_reversals", "mean_motion_tool_calls",
        "target_filtered_out_rate", "max_ticks_rate",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for key in sorted(buckets):
            rs = buckets[key]
            succ = [r for r in rs if int(r["success"]) == 1]
            times = [float(r["completion_time_sec"]) for r in succ]
            inputs = [int(r["total_inputs"]) for r in succ]
            w.writerow({
                "mode":       key[0],
                "condition":  key[1],
                "difficulty": key[2],
                "n_rollouts": len(rs),
                "success_rate": round(len(succ) / len(rs), 4) if rs else 0.0,
                "mean_completion_time_sec": round(_safe_mean(times), 3),
                "std_completion_time_sec":  round(_safe_std(times), 3),
                "mean_total_inputs":        round(_safe_mean(inputs), 2),
                "std_total_inputs":         round(_safe_std(inputs), 2),
                "mean_interactions":        round(_safe_mean([int(r["interactions"]) for r in rs]), 2),
                "mean_dropped_inputs":      round(_safe_mean([int(r["dropped_inputs"]) for r in rs]), 2),
                "mean_direction_reversals": round(_safe_mean([int(r["direction_reversals"]) for r in rs]), 2),
                "mean_motion_tool_calls":   round(_safe_mean([int(r["motion_tool_calls"]) for r in rs]), 2),
                "target_filtered_out_rate": round(_safe_mean([int(r["target_filtered_out"]) for r in rs]), 4),
                "max_ticks_rate":           round(_safe_mean([int(r["terminated_at_max_ticks"]) for r in rs]), 4),
            })


# ── main driver ────────────────────────────────────────────────────────────


def _parse_model_paths(raw: List[str]) -> List[Tuple[str, str]]:
    """Parse ``--model_paths name=path`` entries.

    Accepts either "name=path" or bare "path" (name derived from basename).
    """
    out: List[Tuple[str, str]] = []
    for entry in raw:
        if "=" in entry:
            name, path = entry.split("=", 1)
        else:
            path = entry
            name = Path(path).name
        out.append((name.strip(), path.strip()))
    return out


def _run_sweep_for_model(
    scenarios,
    scenarios_path: str,
    model_key: str,
    model_path: Optional[str],
    args,
    out_dir: Path,
) -> None:
    """Run the full (scenarios × conditions × modes × seeds) sweep for one model.

    For non-hf_ft backends, the model_key is informational only.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: List[Dict[str, Any]] = []
    for s in scenarios:
        for cond in args.conditions:
            for mode in args.modes:
                for seed in range(args.n_seeds):
                    jobs.append({
                        "scenarios_path": scenarios_path,
                        "scenario_id":    s.scenario_id,
                        "mode":           mode,
                        "condition":      cond,
                        "seed":           seed,
                        "backend":        args.backend,
                        "model_path":     model_path,
                        "heuristic_name": args.heuristic,
                        "max_ticks":      args.max_ticks,
                        "use_per_scenario_priors": not args.no_priors,
                    })

    total = len(jobs)
    print(f"[sweep:{model_key}] {len(scenarios)} scenarios × {len(args.conditions)} conditions × "
          f"{len(args.modes)} modes × {args.n_seeds} seeds = {total} rollouts")
    print(f"[sweep:{model_key}] backend={args.backend!r}  workers={args.workers}  priors_calibrated={not args.no_priors}")

    rollouts_csv = out_dir / "rollouts.csv"
    t0 = time.time()
    with rollouts_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROLLOUT_FIELDS)
        w.writeheader()

        if args.backend == "hf_ft" or args.workers <= 1:
            # Single-process: required for hf_ft (model lives in main process VRAM).
            for i, job in enumerate(jobs, 1):
                row = _run_one(job)
                w.writerow(row)
                f.flush()
                if i % args.progress_every == 0 or i == total:
                    elapsed = time.time() - t0
                    rate = i / max(elapsed, 1e-6)
                    eta = (total - i) / max(rate, 1e-6)
                    print(f"  [{model_key} {i}/{total}]  {rate:.1f} roll/s  ETA {eta/60:.1f} min")
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futures = [ex.submit(_run_one, job) for job in jobs]
                for i, fut in enumerate(as_completed(futures), 1):
                    row = fut.result()
                    w.writerow(row)
                    f.flush()
                    if i % args.progress_every == 0 or i == total:
                        elapsed = time.time() - t0
                        rate = i / max(elapsed, 1e-6)
                        eta = (total - i) / max(rate, 1e-6)
                        print(f"  [{model_key} {i}/{total}]  {rate:.1f} roll/s  ETA {eta/60:.1f} min")

    elapsed = time.time() - t0
    print(f"[sweep:{model_key}] wrote {total} rollouts to {rollouts_csv} in {elapsed/60:.1f} min")

    by_condition_csv = out_dir / "by_condition.csv"
    aggregate(rollouts_csv, by_condition_csv)
    print(f"[sweep:{model_key}] aggregated → {by_condition_csv}")

    meta = {
        "model_key":      model_key,
        "model_path":     model_path,
        "config":         vars(args),
        "n_scenarios":    len(scenarios),
        "total_rollouts": total,
        "elapsed_sec":    round(elapsed, 1),
        "rollouts_csv":   str(rollouts_csv),
        "by_condition_csv": str(by_condition_csv),
    }
    (out_dir / "sweep_meta.json").write_text(json.dumps(meta, indent=2))


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", required=True, help="Input scenarios.labeled.jsonl")
    ap.add_argument("--out_dir", required=True, help="Output directory (one subdir per model)")
    ap.add_argument("--modes", nargs="+", default=["manual", "prime"],
                     choices=["manual", "prime"])
    ap.add_argument("--backend", default="oracle",
                     choices=["manual", "defer", "oracle", "hf_ft", "heuristic"],
                     help="Backend used in PRIME mode (manual mode ignores this)")
    ap.add_argument("--model_paths", nargs="*", default=[],
                     help="One or more 'name=path' entries (only used with --backend hf_ft). "
                          "Bare paths are accepted; the name is derived from the basename. "
                          "Each model produces its own out_dir/<name>/ subdir.")
    ap.add_argument("--heuristic", default=None, choices=list(HEURISTIC_REGISTRY),
                     help="Heuristic decision rule (only used with --backend heuristic). "
                          "Output lands in out_dir/<heuristic>/ when set.")
    ap.add_argument("--conditions", nargs="+", default=[p.name for p in STANDARD_CONDITIONS],
                     choices=[p.name for p in STANDARD_CONDITIONS])
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--max_ticks", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=1,
                     help="Parallel worker processes. Use 1 for hf_ft (GPU). "
                          "Manual / defer / oracle scale linearly with workers.")
    ap.add_argument("--max_scenarios", type=int, default=0,
                     help="0 = all; otherwise process only the first N scenarios (smoke tests)")
    ap.add_argument("--no_priors", action="store_true",
                     help="Disable per-scenario priors calibration (fallback to defaults)")
    ap.add_argument("--progress_every", type=int, default=200,
                     help="Print progress every N completed rollouts")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios_path = str(Path(args.scenarios).resolve())

    scenarios = load_scenarios(scenarios_path)
    if args.max_scenarios > 0:
        scenarios = scenarios[: args.max_scenarios]

    # Decide what models to sweep over. Only hf_ft needs an actual list;
    # heuristic uses --heuristic for naming; other backends run once with key="-".
    if args.backend == "hf_ft":
        if not args.model_paths:
            ap.error("--backend hf_ft requires --model_paths")
        model_pairs = _parse_model_paths(args.model_paths)
    elif args.backend == "heuristic":
        if not args.heuristic:
            ap.error("--backend heuristic requires --heuristic")
        model_pairs = [(args.heuristic, None)]
    else:
        model_pairs = [("-", None)]

    print(f"[sweep] backend={args.backend!r}  n_models={len(model_pairs)}  "
          f"out_dir={out_dir}")
    for model_key, model_path in model_pairs:
        sub = out_dir / model_key if len(model_pairs) > 1 or args.backend in ("hf_ft", "heuristic") else out_dir
        _run_sweep_for_model(
            scenarios=scenarios,
            scenarios_path=scenarios_path,
            model_key=model_key,
            model_path=model_path,
            args=args,
            out_dir=sub,
        )
        # Free VRAM between models so the next load doesn't OOM.
        if args.backend == "hf_ft":
            _free_hf_cache()


if __name__ == "__main__":
    main()
