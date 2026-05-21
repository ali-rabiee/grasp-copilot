"""Reranker ablation sweep — thin shim over scenario_noise_sweep.

Per plan §4. Reuses load_scenarios, NoiseProfile, run_rollout, and the
existing backend builders from scenario_noise_sweep; adds a --rerank_mode
knob plus a DialogLogger that captures every emitted INTERACT for the
offline IG analysis (§5).

CLI (typical):
    python -m evaluation.reranker.run_reranker_sweep \\
        --scenarios evaluation/results/robustness/user_input_noise/scenarios/scenarios.labeled.jsonl \\
        --out_dir   evaluation/results/reranker/ablation/oracle_woz_lora__info_gain \\
        --backend   hf_ft \\
        --model_paths oracle_woz_lora=models/qwen2_5_3b_oracle_woz_lora \\
        --modes     prime \\
        --conditions clean compound_mid \\
        --n_seeds   5 \\
        --rerank_mode info_gain \\
        --dialog_log evaluation/results/reranker/ablation/oracle_woz_lora__info_gain/dialogs.jsonl
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from evaluation import _bootstrap  # noqa: F401
except Exception:
    import _bootstrap  # type: ignore  # noqa: F401

from evaluation.benchmarks.scenario_noise_sweep import (
    CONDITION_LOOKUP,
    HEURISTIC_REGISTRY,
    ROLLOUT_FIELDS,
    OracleBackend,
    _build_heuristic_backend,
    _free_hf_cache,
    _parse_model_paths,
    aggregate,
)
from evaluation.rollouts.noise import STANDARD_CONDITIONS
from evaluation.rollouts.rollout_loop import run_rollout
from evaluation.scenarios import load_scenarios
from evaluation.reranker.dialog_logger import DialogLogger
from llm.reranker.policy_wrapper import RerankerConfig, make_reranked_backend


# ── HF model cache (separate from scenario_noise_sweep's — we keep handles) ─

_HF: Dict[str, Any] = {}


def _ensure_hf_loaded(model_path: str) -> Tuple[Any, Any, Any]:
    """Return (model, tok, base_cfg). Loads once per process, cached by path."""
    if _HF.get("path") == model_path and "model" in _HF:
        return _HF["model"], _HF["tok"], _HF["cfg"]
    from llm.inference import (
        InferenceConfig,
        _load_model_and_tokenizer,
    )
    cfg = InferenceConfig(
        model_path=model_path,
        use_4bit=False,
        temperature=0.0, top_p=1.0,
        max_new_tokens=256,
        seed=0, deterministic=True,
    )
    print(f"[rerank-sweep] loading HF model from {model_path}...", flush=True)
    t0 = time.time()
    model, tok = _load_model_and_tokenizer(cfg)
    print(f"[rerank-sweep] model loaded in {time.time() - t0:.1f}s", flush=True)
    _HF.clear()
    _HF["model"], _HF["tok"], _HF["cfg"], _HF["path"] = model, tok, cfg, model_path
    return model, tok, cfg


def _free_hf() -> None:
    _HF.clear()
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _build_inner_hf_backend(model, tok, cfg) -> Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]:
    from llm.inference import _build_messages, _generate_once
    from evaluation.benchmarks.offline_exec_benchmark import _parse_model_json
    from llm.reranker.candidates import truncate_past_dialogs

    instruction = (
        "Given the robot observation and dialog context, infer the user's intent and "
        "emit exactly one tool call. Output ONLY the tool call JSON with keys tool and args. "
        "If the tool is INTERACT, you must output at most 5 choices total."
    )

    def backend(input_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Cap past_dialogs in the prompt — mirrors the noise-sweep runaway
        # fix (commit eeb8ee2): without truncation, dialog loops grow the
        # prompt past 32k context and the model decodes nonsense for the
        # full max_new_tokens budget, turning each call into minutes.
        truncated = truncate_past_dialogs(input_dict)
        prompt = f"{instruction}\n\nInput:\n{json.dumps(truncated, ensure_ascii=False)}"
        try:
            raw = _generate_once(model, tok, _build_messages(prompt), cfg)
            pred_obj, _ = _parse_model_json(raw)
            return pred_obj
        except Exception:
            return None

    return backend


def _build_backend(
    *, backend_name: str, target_obj_id: str,
    model_path: Optional[str], heuristic_name: Optional[str],
    rerank_cfg: RerankerConfig, dialog_log: Optional[DialogLogger],
) -> Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]]:
    """Construct (and optionally wrap) the per-rollout backend.

    `dialog_log` is shared across rollouts; its set_episode() must be
    called by the caller before this backend is used.
    """
    inner: Optional[Callable]
    model = tok = base_cfg = None

    if backend_name in ("manual", "defer"):
        return None
    if backend_name == "oracle":
        inner = OracleBackend(target_obj_id=target_obj_id)
    elif backend_name == "heuristic":
        inner = _build_heuristic_backend(heuristic_name)
    elif backend_name == "hf_ft":
        if not model_path:
            raise ValueError("hf_ft backend requires --model_paths")
        model, tok, base_cfg = _ensure_hf_loaded(model_path)
        inner = _build_inner_hf_backend(model, tok, base_cfg)
    else:
        raise ValueError(f"unknown backend {backend_name!r}")

    # No reranking → return inner directly (preserves OracleBackend's
    # .state / .on_user_reply attributes, which the rollout loop touches).
    if rerank_cfg.mode == "none" and dialog_log is None:
        return inner

    # When wrapping a stateful inner (OracleBackend), forward the .state
    # attribute and on_user_reply so the rollout loop's
    # `getattr(backend, "state", None)` and on_reply paths still work.
    wrapped = make_reranked_backend(
        inner, model=model, tok=tok, base_cfg=base_cfg,
        config=rerank_cfg, dialog_log=dialog_log,
    )

    # Splice state/on_user_reply through for stateful inner backends.
    if hasattr(inner, "state"):
        try:
            wrapped.state = inner.state  # type: ignore[attr-defined]
        except Exception:
            pass
    if hasattr(inner, "on_user_reply"):
        try:
            wrapped.on_user_reply = inner.on_user_reply  # type: ignore[attr-defined]
        except Exception:
            pass
    return wrapped


# ── per-rollout work ───────────────────────────────────────────────────────


def _run_one(
    *, scenario, mode: str, condition: str, seed: int,
    backend_name: str, model_path: Optional[str], heuristic_name: Optional[str],
    max_ticks: int, use_per_scenario_priors: bool,
    rerank_cfg: RerankerConfig, dialog_log: Optional[DialogLogger],
) -> Dict[str, Any]:
    if dialog_log is not None:
        dialog_log.set_episode(scenario.scenario_id, seed, condition)

    # Bump tick counter via a wrapper so the logger's tick matches the rollout.
    backend = _build_backend(
        backend_name="manual" if mode == "manual" else backend_name,
        target_obj_id=scenario.target_obj_id,
        model_path=model_path, heuristic_name=heuristic_name,
        rerank_cfg=rerank_cfg, dialog_log=dialog_log,
    )
    if backend is not None and dialog_log is not None:
        orig = backend
        def tick_wrap(input_dict):
            out = orig(input_dict)
            dialog_log.tick_inc()
            return out
        # Forward state/on_user_reply for stateful backends.
        if hasattr(orig, "state"):
            tick_wrap.state = orig.state  # type: ignore[attr-defined]
        if hasattr(orig, "on_user_reply"):
            tick_wrap.on_user_reply = orig.on_user_reply  # type: ignore[attr-defined]
        backend = tick_wrap

    profile = CONDITION_LOOKUP[condition]
    r = run_rollout(
        scenario, mode=mode, noise_profile=profile, seed=seed,
        backend=backend, max_ticks=max_ticks,
        use_per_scenario_priors=use_per_scenario_priors,
    )
    return {
        "scenario_id":   scenario.scenario_id,
        "subject":       scenario.source.subject,
        "difficulty":    scenario.source.difficulty,
        "trial_mode":    scenario.source.mode,
        "mode":          r.mode,
        "condition":     r.condition,
        "seed":          r.seed,
        "success":       int(r.success),
        "completion_time_sec":     round(r.completion_time_sec, 4),
        "total_inputs":            r.total_inputs,
        "interactions":            r.interactions,
        "motion_tool_calls":       r.motion_tool_calls,
        "mode_switches":           r.mode_switches,
        "direction_reversals":     r.direction_reversals,
        "dropped_inputs":          r.dropped_inputs,
        "selection_perturbations": r.selection_perturbations,
        "direction_perturbations": r.direction_perturbations,
        "target_filtered_out":     int(r.target_filtered_out),
        "terminated_at_max_ticks": int(r.terminated_at_max_ticks),
        "end_reason":              r.end_reason,
    }


# ── main driver ────────────────────────────────────────────────────────────


def _run_sweep_for_model(
    *, scenarios, model_key: str, model_path: Optional[str], args,
    out_dir: Path, rerank_cfg: RerankerConfig,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts_csv = out_dir / "rollouts.csv"

    # Dialog log path — overrideable via --dialog_log, else out_dir/dialogs.jsonl.
    dialog_path = Path(args.dialog_log) if args.dialog_log else out_dir / "dialogs.jsonl"
    # Truncate previous log so this sweep starts clean.
    if dialog_path.exists():
        dialog_path.unlink()
    log_enabled = args.backend in ("hf_ft", "oracle", "heuristic") and "prime" in args.modes

    total_jobs = (
        len(scenarios) * len(args.conditions) * len(args.modes) * args.n_seeds
    )
    print(f"[rerank-sweep:{model_key}] {len(scenarios)} scenarios × "
          f"{len(args.conditions)} conditions × {len(args.modes)} modes × "
          f"{args.n_seeds} seeds = {total_jobs} rollouts", flush=True)
    print(f"[rerank-sweep:{model_key}] backend={args.backend!r} rerank_mode={rerank_cfg.mode!r} "
          f"k={rerank_cfg.k} temp={rerank_cfg.temperature} prior={rerank_cfg.prior}", flush=True)
    print(f"[rerank-sweep:{model_key}] dialog_log → {dialog_path}", flush=True)

    t0 = time.time()
    i = 0
    with rollouts_csv.open("w", newline="") as fcsv, \
         (DialogLogger(path=dialog_path) if log_enabled else _NullLogger()) as dlog:
        w = csv.DictWriter(fcsv, fieldnames=ROLLOUT_FIELDS)
        w.writeheader()
        for s in scenarios:
            for cond in args.conditions:
                for mode in args.modes:
                    for seed in range(args.n_seeds):
                        row = _run_one(
                            scenario=s, mode=mode, condition=cond, seed=seed,
                            backend_name=args.backend,
                            model_path=model_path, heuristic_name=args.heuristic,
                            max_ticks=args.max_ticks,
                            use_per_scenario_priors=not args.no_priors,
                            rerank_cfg=rerank_cfg,
                            dialog_log=dlog if log_enabled and mode == "prime" else None,
                        )
                        w.writerow(row)
                        fcsv.flush()
                        i += 1
                        if i % args.progress_every == 0 or i == total_jobs:
                            elapsed = time.time() - t0
                            rate = i / max(elapsed, 1e-6)
                            eta = (total_jobs - i) / max(rate, 1e-6)
                            print(f"  [{model_key} {i}/{total_jobs}] {rate:.1f} roll/s "
                                  f"ETA {eta/60:.1f} min", flush=True)

    elapsed = time.time() - t0
    print(f"[rerank-sweep:{model_key}] wrote {total_jobs} rollouts to {rollouts_csv} "
          f"in {elapsed/60:.1f} min", flush=True)

    by_condition_csv = out_dir / "by_condition.csv"
    aggregate(rollouts_csv, by_condition_csv)
    print(f"[rerank-sweep:{model_key}] aggregated → {by_condition_csv}", flush=True)

    meta = {
        "model_key":      model_key,
        "model_path":     model_path,
        "config":         vars(args),
        "rerank":         {"mode": rerank_cfg.mode, "k": rerank_cfg.k,
                            "temperature": rerank_cfg.temperature,
                            "top_p": rerank_cfg.top_p, "prior": rerank_cfg.prior,
                            "seed": rerank_cfg.seed},
        "n_scenarios":    len(scenarios),
        "total_rollouts": total_jobs,
        "elapsed_sec":    round(elapsed, 1),
        "rollouts_csv":   str(rollouts_csv),
        "by_condition_csv": str(by_condition_csv),
        "dialog_log":     str(dialog_path) if log_enabled else None,
    }
    (out_dir / "sweep_meta.json").write_text(json.dumps(meta, indent=2))


class _NullLogger:
    """Used when dialog logging is disabled (manual/defer modes)."""
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def set_episode(self, *a, **kw): pass
    def tick_inc(self): pass


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--modes", nargs="+", default=["prime"], choices=["manual", "prime"])
    ap.add_argument("--backend", default="hf_ft",
                    choices=["manual", "defer", "oracle", "hf_ft", "heuristic"])
    ap.add_argument("--model_paths", nargs="*", default=[],
                    help="One or more 'name=path' entries (only used with --backend hf_ft).")
    ap.add_argument("--heuristic", default=None, choices=list(HEURISTIC_REGISTRY))
    ap.add_argument("--conditions", nargs="+", default=[p.name for p in STANDARD_CONDITIONS],
                    choices=[p.name for p in STANDARD_CONDITIONS])
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--max_ticks", type=int, default=2000)
    ap.add_argument("--max_scenarios", type=int, default=0)
    ap.add_argument("--no_priors", action="store_true")
    ap.add_argument("--progress_every", type=int, default=100)
    # Reranker knobs.
    ap.add_argument("--rerank_mode", default="info_gain",
                    choices=["none", "info_gain", "random", "oracle"])
    ap.add_argument("--k_candidates", type=int, default=5)
    ap.add_argument("--rerank_temperature", type=float, default=0.7)
    ap.add_argument("--rerank_top_p", type=float, default=0.95)
    ap.add_argument("--prior", default="uniform", choices=["uniform", "motion_weighted"])
    ap.add_argument("--rerank_seed", type=int, default=0)
    ap.add_argument("--dialog_log", default=None,
                    help="Optional path for the dialogs.jsonl; defaults to <out_dir>/dialogs.jsonl")
    args = ap.parse_args(argv)

    rerank_cfg = RerankerConfig(
        mode=args.rerank_mode,
        k=int(args.k_candidates),
        temperature=float(args.rerank_temperature),
        top_p=float(args.rerank_top_p),
        prior=args.prior,
        seed=int(args.rerank_seed),
    )

    scenarios = load_scenarios(str(Path(args.scenarios).resolve()))
    if args.max_scenarios > 0:
        scenarios = scenarios[: args.max_scenarios]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "hf_ft":
        if not args.model_paths:
            ap.error("--backend hf_ft requires --model_paths")
        model_pairs = _parse_model_paths(args.model_paths)
    elif args.backend == "heuristic":
        if not args.heuristic:
            ap.error("--backend heuristic requires --heuristic")
        model_pairs = [(args.heuristic, None)]
    else:
        model_pairs = [(args.backend, None)]

    for model_key, model_path in model_pairs:
        sub = out_dir / model_key if len(model_pairs) > 1 else out_dir
        _run_sweep_for_model(
            scenarios=scenarios,
            model_key=model_key, model_path=model_path,
            args=args, out_dir=sub, rerank_cfg=rerank_cfg,
        )
        if args.backend == "hf_ft":
            _free_hf()


if __name__ == "__main__":
    main()
