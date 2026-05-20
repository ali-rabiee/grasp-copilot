"""
Robustness sweep: every trained LoRA × every env × noise levels.

Loads each model once and sweeps it across:
    envs = {ycb, stacking, pouring}  (mapped to the per-env oracle valid sets)
    perturbation = user_input (jitter gripper history cells + yaw bins)
    noise_levels = {0.0, 0.1, 0.2, 0.3, 0.5}

Per-(model, env, p) results are cached as CSV rows so re-runs skip cells that
already exist (matched on model_safe, env, noise_level). Heuristic baselines
(Oracle, H1) are evaluated once per env and reused across models in the plot.

Outputs:
    evaluation/eval_outputs/paper_benchmark/robustness/sweep.csv
    evaluation/eval_outputs/paper_benchmark/robustness/sweep_aggregated.json
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from . import _bootstrap  # noqa: F401
except Exception:
    import _bootstrap  # type: ignore  # noqa: F401

from evaluation.offline_exec_benchmark import (
    _heuristic_ask_if_ambiguous,
    _iter_jsonl,
    _normalize_tool_call,
    _parse_model_json,
    _rate,
)
from evaluation.robustness_benchmark import (
    PERTURBATION_REGISTRY,
    _reconstruct_oracle_state,
    _run_oracle,
    _fresh_counters,
    _score,
)
from evaluation.run_paper_benchmark import TRAINED_MODELS, ModelEntry, REPO_ROOT
from llm.utils import json_loads_strict, set_seed


# =============================================================================
# Configuration
# =============================================================================

ENV_TO_CONTRACT = {
    "ycb":      "data/oracle_valid_ycb/llm_contract_200.jsonl",
    "stacking": "data/oracle_valid_stacking/llm_contract_200.jsonl",
    "pouring":  "data/oracle_valid_pouring/llm_contract_200.jsonl",
}

DEFAULT_NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5]
DEFAULT_PERTURBATION = "user_input"


@dataclass(frozen=True)
class SweepCell:
    model_safe: str
    model_display: str
    env: str
    perturbation: str
    noise_level: float
    n: int
    tool_acc: float
    strict_match: float
    motion_obj_acc: float
    elapsed_s: float


SWEEP_COLUMNS = [
    "model_safe", "model_display", "env", "perturbation",
    "noise_level", "n", "tool_acc", "strict_match", "motion_obj_acc", "elapsed_s",
]


# =============================================================================
# Sweep core
# =============================================================================

def _eval_one_cell(
    model,
    tok,
    cfg,
    rows: List[Dict[str, Any]],
    *,
    perturbation: str,
    noise_level: float,
    seed: int,
    max_examples: int,
    progress_every: int,
    model_safe: str,
    env: str,
    include_oracle: bool,
    include_h1: bool,
) -> Tuple[Dict[str, Dict[str, int]], int]:
    """Evaluate {LLM, Oracle?, H1?} on one (perturbation, noise_level) cell.
    Returns dict of system_name → counters, plus the sample size used."""
    perturb_fn = PERTURBATION_REGISTRY[perturbation]
    rng_run = random.Random(seed)
    n = min(int(max_examples), len(rows)) if int(max_examples) > 0 else len(rows)
    sample = rows[:n]

    systems: Dict[str, Dict[str, int]] = {model_safe: _fresh_counters()}
    if include_oracle:
        systems["Oracle"] = _fresh_counters()
    if include_h1:
        systems["H1"] = _fresh_counters()

    from llm.inference import _build_messages, _generate_once

    for idx, row in enumerate(sample):
        instruction = str(row.get("instruction", ""))
        input_str = str(row.get("input", ""))
        gt_str = str(row.get("output", ""))
        try:
            gt = _normalize_tool_call(json_loads_strict(gt_str))
            inp = json_loads_strict(input_str)
        except Exception:
            continue

        perturbed = perturb_fn(inp, noise_level, rng_run)
        p_str = json.dumps(perturbed, ensure_ascii=False)

        if include_oracle:
            try:
                ostate = _reconstruct_oracle_state(inp, gt)
                oracle_out = _run_oracle(perturbed, ostate)
                if oracle_out:
                    _score(gt, _normalize_tool_call(oracle_out), systems["Oracle"])
                else:
                    systems["Oracle"]["n"] += 1
            except Exception:
                systems["Oracle"]["n"] += 1

        if include_h1:
            try:
                h1 = _normalize_tool_call(_heuristic_ask_if_ambiguous(p_str))
                _score(gt, h1, systems["H1"])
            except Exception:
                systems["H1"]["n"] += 1

        # Model
        prompt = f"{instruction}\n\nInput:\n{p_str}"
        try:
            raw = _generate_once(model, tok, _build_messages(prompt), cfg)
            pred_obj, _ = _parse_model_json(raw)
            if pred_obj:
                _score(gt, _normalize_tool_call(pred_obj), systems[model_safe])
            else:
                systems[model_safe]["n"] += 1
        except Exception:
            systems[model_safe]["n"] += 1

        if progress_every and (idx + 1) % progress_every == 0:
            print(f"    [{model_safe} {env} p={noise_level}] {idx+1}/{len(sample)}")

    return systems, n


# =============================================================================
# Caching
# =============================================================================

def _load_cache(csv_path: Path) -> Dict[Tuple[str, str, float], Dict[str, Any]]:
    cache: Dict[Tuple[str, str, float], Dict[str, Any]] = {}
    if not csv_path.exists():
        return cache
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                key = (row["model_safe"], row["env"], float(row["noise_level"]))
                cache[key] = row
            except Exception:
                continue
    return cache


def _append_rows(csv_path: Path, new_rows: List[Dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    first = not csv_path.exists()
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SWEEP_COLUMNS)
        if first:
            w.writeheader()
        for r in new_rows:
            w.writerow({k: r.get(k) for k in SWEEP_COLUMNS})


# =============================================================================
# Orchestration
# =============================================================================

def run(
    *,
    models: List[ModelEntry],
    envs: List[str],
    perturbation: str,
    noise_levels: List[float],
    max_examples: int,
    seed: int,
    out_dir: Path,
    rerun: bool,
    progress_every: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep.csv"
    cache = _load_cache(csv_path) if not rerun else {}

    # Cache eval set rows (we'll reuse across noise levels).
    env_rows: Dict[str, List[Dict[str, Any]]] = {}
    for env in envs:
        p = ENV_TO_CONTRACT.get(env)
        if not p:
            raise SystemExit(f"Unknown env: {env}")
        full = (REPO_ROOT / p) if not Path(p).is_absolute() else Path(p)
        env_rows[env] = list(_iter_jsonl(str(full)))
        print(f"[sweep] {env}: {len(env_rows[env])} rows from {full}")

    set_seed(int(seed))

    from llm.inference import InferenceConfig, _load_model_and_tokenizer

    for mi, model in enumerate(models, start=1):
        if model.kind != "llm":
            continue
        print(f"\n{'#'*72}\n# [{mi}/{len(models)}] {model.display}\n{'#'*72}")

        # Check whether all cells for this model are already cached.
        all_cached = all(
            (model.safe_name, env, p) in cache
            for env in envs for p in noise_levels
        )
        if all_cached and not rerun:
            print(f"  [cached] all cells present for {model.safe_name}")
            continue

        cfg = InferenceConfig(
            model_path=str(model.model_path),
            use_4bit=False,
            temperature=0.0, top_p=1.0, max_new_tokens=256,
            seed=int(seed), deterministic=True,
        )
        t0 = time.time()
        mdl, tok = _load_model_and_tokenizer(cfg)
        print(f"  loaded in {time.time()-t0:.1f}s")

        new_rows: List[Dict[str, Any]] = []
        for env in envs:
            rows = env_rows[env]
            for p in noise_levels:
                key = (model.safe_name, env, p)
                if key in cache and not rerun:
                    print(f"  [cached] {model.safe_name} env={env} p={p}")
                    continue
                # Run Oracle/H1 only at p=0 and the smallest LLM in the run
                # (they don't depend on the LLM weights — we capture them once
                # per env/noise via a separate pass below). Here, skip for speed.
                tc0 = time.time()
                systems, n_used = _eval_one_cell(
                    mdl, tok, cfg, rows,
                    perturbation=perturbation,
                    noise_level=p,
                    seed=seed,
                    max_examples=max_examples,
                    progress_every=progress_every,
                    model_safe=model.safe_name,
                    env=env,
                    include_oracle=False,
                    include_h1=False,
                )
                elapsed = time.time() - tc0
                counters = systems[model.safe_name]
                new_rows.append({
                    "model_safe":     model.safe_name,
                    "model_display":  model.display,
                    "env":            env,
                    "perturbation":   perturbation,
                    "noise_level":    f"{p:.2f}",
                    "n":              counters["n"],
                    "tool_acc":       f"{_rate(counters['tool_correct'], counters['n']) * 100:.2f}",
                    "strict_match":   f"{_rate(counters['strict_exact'], counters['n']) * 100:.2f}",
                    "motion_obj_acc": f"{_rate(counters['motion_obj_correct'], counters['motion_n']) * 100:.2f}",
                    "elapsed_s":      f"{elapsed:.1f}",
                })
                print(f"  [done] {model.safe_name} env={env} p={p}  tool_acc={counters['tool_correct']}/{counters['n']}={_rate(counters['tool_correct'], counters['n'])*100:.1f}%  ({elapsed:.1f}s)")

        if new_rows:
            _append_rows(csv_path, new_rows)

        # Free GPU memory
        try:
            import gc, torch
            del mdl, tok
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    # Second pass: heuristics + oracle baseline per (env, p), no LLM needed.
    # These are reused as horizontal reference lines in the figures.
    baseline_csv = out_dir / "sweep_baselines.csv"
    if rerun or not baseline_csv.exists():
        print("\n[sweep] computing Oracle/H1 baseline curves (CPU only)...")
        _compute_baselines(env_rows, perturbation, noise_levels, max_examples, seed, baseline_csv, progress_every)

    print(f"\n[sweep] wrote {csv_path}")
    print(f"[sweep] wrote {baseline_csv}")


def _compute_baselines(
    env_rows: Dict[str, List[Dict[str, Any]]],
    perturbation: str,
    noise_levels: List[float],
    max_examples: int,
    seed: int,
    out_path: Path,
    progress_every: int,
) -> None:
    perturb_fn = PERTURBATION_REGISTRY[perturbation]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["system", "env", "perturbation", "noise_level", "n", "tool_acc", "strict_match", "motion_obj_acc"]

    write_rows: List[Dict[str, Any]] = []
    for env, rows in env_rows.items():
        n = min(int(max_examples), len(rows)) if int(max_examples) > 0 else len(rows)
        sample = rows[:n]
        for p in noise_levels:
            rng_run = random.Random(seed)
            oc = _fresh_counters()
            hc = _fresh_counters()
            for row in sample:
                try:
                    gt = _normalize_tool_call(json_loads_strict(str(row.get("output", ""))))
                    inp = json_loads_strict(str(row.get("input", "")))
                except Exception:
                    continue
                perturbed = perturb_fn(inp, p, rng_run)
                p_str = json.dumps(perturbed, ensure_ascii=False)
                try:
                    ostate = _reconstruct_oracle_state(inp, gt)
                    o_out = _run_oracle(perturbed, ostate)
                    if o_out:
                        _score(gt, _normalize_tool_call(o_out), oc)
                    else:
                        oc["n"] += 1
                except Exception:
                    oc["n"] += 1
                try:
                    h1 = _normalize_tool_call(_heuristic_ask_if_ambiguous(p_str))
                    _score(gt, h1, hc)
                except Exception:
                    hc["n"] += 1
            for sys_name, c in (("Oracle", oc), ("H1_ask_if_ambiguous", hc)):
                write_rows.append({
                    "system": sys_name, "env": env, "perturbation": perturbation,
                    "noise_level": f"{p:.2f}", "n": c["n"],
                    "tool_acc":       f"{_rate(c['tool_correct'], c['n']) * 100:.2f}",
                    "strict_match":   f"{_rate(c['strict_exact'], c['n']) * 100:.2f}",
                    "motion_obj_acc": f"{_rate(c['motion_obj_correct'], c['motion_n']) * 100:.2f}",
                })
            print(f"  [baseline] env={env} p={p}  Oracle tool={_rate(oc['tool_correct'], oc['n'])*100:.1f}%  H1 tool={_rate(hc['tool_correct'], hc['n'])*100:.1f}%")

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in write_rows:
            w.writerow(r)


# =============================================================================
# Plot helper (kept light; the main figure script can read sweep.csv)
# =============================================================================

def plot_curves(out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from evaluation.plots.paper_figures import MODEL_COLORS, ENV_LABEL

    csv_path = out_dir / "sweep.csv"
    base_path = out_dir / "sweep_baselines.csv"
    if not csv_path.exists() and not base_path.exists():
        print(f"[plot] missing both {csv_path} and {base_path}")
        return

    rows: List[Dict[str, str]] = []
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    by_model_env: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    display_of: Dict[str, str] = {}
    for r in rows:
        key = (r["model_safe"], r["env"])
        by_model_env.setdefault(key, []).append((float(r["noise_level"]), float(r["tool_acc"])))
        display_of[r["model_safe"]] = r["model_display"]
    baselines: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    if base_path.exists():
        with open(base_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                baselines.setdefault((r["system"], r["env"]), []).append(
                    (float(r["noise_level"]), float(r["tool_acc"])))

    envs = sorted({k[1] for k in by_model_env} | {k[1] for k in baselines})
    if not envs:
        print("[plot] no env data to plot")
        return
    fig, axes = plt.subplots(1, len(envs), figsize=(4.0 * len(envs), 3.6), sharey=True)
    if len(envs) == 1:
        axes = [axes]

    for ai, env in enumerate(envs):
        ax = axes[ai]
        for (m, e), pts in by_model_env.items():
            if e != env:
                continue
            pts = sorted(pts)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, "-o", color=MODEL_COLORS.get(m, "#666"), lw=1.7, ms=5,
                    label=display_of.get(m, m))
        # baselines
        for sys_name, color, ls in (("Oracle", "#666", "--"), ("H1_ask_if_ambiguous", "#888", ":")):
            pts = baselines.get((sys_name, env), [])
            if pts:
                pts = sorted(pts)
                ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, color=color, lw=1.2,
                        label=sys_name if ai == 0 else None)
        ax.set_xlabel("User-input noise level $p$")
        ax.set_title(ENV_LABEL.get(env, env))
        all_levels = [float(r["noise_level"]) for r in rows] + [
            p for k, pts in baselines.items() for (p, _) in pts
        ]
        ax.set_xlim(-0.02, (max(all_levels) if all_levels else 0.5) + 0.02)
        ax.set_ylim(0, 100)
        if ai == 0:
            ax.set_ylabel("Tool-call accuracy (%)")
    fig.suptitle("Robustness to teleoperation noise", y=1.02)
    axes[0].legend(loc="lower left", fontsize=7.5, frameon=True, ncol=1)
    out_pdf = out_dir / "robustness_curves.pdf"
    out_png = out_dir / "robustness_curves.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_pdf} / {out_png}")


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default="evaluation/eval_outputs/paper_benchmark/robustness")
    ap.add_argument("--models", default=None, help="Comma-separated safe_names; default = all trained")
    ap.add_argument("--envs", default=None, help="Comma-separated env keys (ycb,stacking,pouring)")
    ap.add_argument("--perturbation", default=DEFAULT_PERTURBATION, choices=list(PERTURBATION_REGISTRY))
    ap.add_argument("--noise_levels", default=None, help="Comma-separated floats; default 0.0,0.1,0.2,0.3,0.5")
    ap.add_argument("--max_examples", type=int, default=300, help="Per-cell sample size cap")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--progress_every", type=int, default=200)
    ap.add_argument("--plot_only", action="store_true", help="Skip evaluation; just regenerate the plot")
    ap.add_argument("--baselines_only", action="store_true",
                    help="Skip LLM models; compute only Oracle/H1 baseline curves (CPU only, fast)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    if args.plot_only:
        plot_curves(out_dir)
        return

    if args.baselines_only:
        envs = list(ENV_TO_CONTRACT.keys())
        if args.envs:
            keys = {n.strip() for n in args.envs.split(",") if n.strip()}
            envs = [e for e in envs if e in keys]
        levels = DEFAULT_NOISE_LEVELS
        if args.noise_levels:
            levels = [float(x) for x in args.noise_levels.split(",")]

        env_rows: Dict[str, List[Dict[str, Any]]] = {}
        for env in envs:
            p = ENV_TO_CONTRACT[env]
            full = (REPO_ROOT / p) if not Path(p).is_absolute() else Path(p)
            env_rows[env] = list(_iter_jsonl(str(full)))
            print(f"[baselines] {env}: {len(env_rows[env])} rows from {full}")

        out_dir.mkdir(parents=True, exist_ok=True)
        baseline_csv = out_dir / "sweep_baselines.csv"
        _compute_baselines(env_rows, args.perturbation, levels,
                           int(args.max_examples), int(args.seed),
                           baseline_csv, int(args.progress_every))
        print(f"[baselines] wrote {baseline_csv}")
        return

    models = list(TRAINED_MODELS)
    if args.models:
        keys = {n.strip() for n in args.models.split(",") if n.strip()}
        models = [m for m in models if m.safe_name in keys]
    envs = list(ENV_TO_CONTRACT.keys())
    if args.envs:
        keys = {n.strip() for n in args.envs.split(",") if n.strip()}
        envs = [e for e in envs if e in keys]
    levels = DEFAULT_NOISE_LEVELS
    if args.noise_levels:
        levels = [float(x) for x in args.noise_levels.split(",")]

    print(f"[sweep] models={[m.safe_name for m in models]}")
    print(f"[sweep] envs={envs}")
    print(f"[sweep] noise_levels={levels}")
    print(f"[sweep] max_examples={args.max_examples}")
    print(f"[sweep] out_dir={out_dir}")

    run(
        models=models,
        envs=envs,
        perturbation=args.perturbation,
        noise_levels=levels,
        max_examples=int(args.max_examples),
        seed=int(args.seed),
        out_dir=out_dir,
        rerun=bool(args.rerun),
        progress_every=int(args.progress_every),
    )
    plot_curves(out_dir)


if __name__ == "__main__":
    main()
