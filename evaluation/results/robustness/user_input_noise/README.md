# User-input-noise robustness (new — episode-level rollouts)

Episode-level noise-robustness sweep. Seeds the lightweight 3×3 grid sim
with **real PRIME_LOGS user-study scenarios** and runs forward rollouts
through a **priors-calibrated scripted user** under four noise channels
modeling low-bandwidth interfaces (joystick / head-array / BCI / EMG).

This is the experiment that defends the paper's central claim
*"PRIME's advantage grows as user input bandwidth shrinks."*

```
user_input_noise/
├── README.md            ← you are here
├── scenarios/           160 real-user scenarios (corpus, committed to repo)
│   ├── scenarios.jsonl           ← extractor output (raw, with 124 unlabeled targets)
│   ├── scenarios.labeled.jsonl   ← 160/160 labeled (36 auto + 124 video-verified)
│   ├── scenarios_contract.jsonl  ← same scenarios in contract-JSONL format
│   └── scenarios_summary.json    ← counts, provenance, scene templates
└── sweeps/              per-run sweep outputs (NOT committed — recomputable)
    └── <run_name>/
        ├── rollouts.csv          (one row per scenario × mode × condition × seed)
        ├── by_condition.csv      (aggregated mean / std / success-rate)
        └── sweep_meta.json       (run config + timing)
```

---

## The experiment in one diagram

```
real user-study trial  ──► extract scenario (initial layout + target + priors)
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │  Episode.from_scenario  │   3×3 grid sim
                        └─────────────────────────┘
                                     │
   ┌────────────────┐  cmd  ┌────────┴────────┐  decision  ┌──────────┐
   │ ScriptedUser   ├──────►│  NoiseInjector  ├───────────►│  PRIME   │
   │ (target-aware, │       │  (4 channels)   │            │ backend  │
   │  priors-calib) │       └─────────────────┘            │ (LLM /   │
   └────────────────┘                                       │  oracle /│
                              ▲                             │  hf_ft / │
                              │  reply, selection-flipped   │  manual) │
                              └─────────────────────────────┴──────────┘

                                     │
                                     ▼
   per-rollout metrics: success, completion time, total user inputs,
                        interactions, mode switches, direction reversals,
                        dropped inputs, ...
```

---

## What's in `scenarios/`

160 scenarios extracted from `PRIME_LOGS/{manual,assistive}/s1..s8/{easy,hard}/`:

| Field | Value |
|---|---|
| `n_scenarios` | 160 |
| `target_label_source: tool_call` | 36 (auto-derived from last APPROACH/ALIGN_YAW target_object_id in the trial) |
| `target_label_source: hand_label` | 124 (video-verified by Ali) |
| `layout_source: state_snapshot` | 36 (real layouts from assistive trials' tool_calls.jsonl) |
| `layout_source: borrowed_template` | 124 (per-(subject,difficulty) canonical scene or pooled fallback; documented in `notes`) |
| `user_priors`  | per-trial: total_commands, mean_active_burst_sec, mode shares, mode-switches/sec, direction-reversals/sec |

The corpus is **environment-restricted to reach-to-grasp YCB** — that is the
only env the real user study covered (see paper Section IV-B). Generalizing
this experiment to stacking / pouring would require collecting real user
trials in those envs.

**`scenarios/scenarios.labeled.jsonl` is the authoritative input** for the
sweep. The contract version (`scenarios_contract.jsonl`) is for plugging into
single-timestep evaluators like `offline_exec_benchmark.py` — not used by the
rollout sweep.

---

## How to run the sweep

The runner is `evaluation/benchmarks/scenario_noise_sweep.py`. It supports
five backends; you typically run several and compare:

| Backend | What it is | When to use it |
|---|---|---|
| `manual` | No PRIME — scripted user drives directly | The baseline. Fast (CPU only). |
| `defer` | PRIME mode that always defers — equivalent to manual but routes through the PRIME loop | Sanity check that PRIME mode is mechanically correct |
| `oracle` | The heuristic oracle from `data_generator/oracle.py` as a stateful PRIME stand-in | Cheap diagnostic (no GPU) |
| `heuristic` | h1/h2/sa1/sa2 from `offline_exec_benchmark` lifted into the rollout loop | "LLM > rules under noise" baseline. Stateless heuristics will tend to dialog-loop, which is honest — heuristics are state-blind by design. |
| `hf_ft` | Fine-tuned Qwen LLM via `llm.inference` | **Headline.** GPU. Use for the paper figures. |

### Local quick-look (manual baseline, 5 seeds)

```bash
python -m evaluation.benchmarks.scenario_noise_sweep \
    --scenarios evaluation/results/robustness/user_input_noise/scenarios/scenarios.labeled.jsonl \
    --out_dir   evaluation/results/robustness/user_input_noise/sweeps/manual \
    --modes manual --backend manual --n_seeds 5 --workers 4
```

Runs ~4800 rollouts in ~3 s on CPU. Already done — see `sweeps/sweep_manual_only/` for an earlier run.

### Unity job (LLM headline + ablation)

```bash
MODEL_KEY=oracle_woz_lora \
MODEL_PATH=models/qwen2_5_3b_oracle_woz_lora \
sbatch unity_config/job_noise_sweep.sbatch
```

See `unity_config/job_noise_sweep.sbatch` for the full Unity recipe and the
list of files you need to transfer to Unity before running it.

---

## How to read the outputs

For each sweep, two CSV files plus a meta JSON:

`rollouts.csv` (per-rollout records — many rows per condition):

```
scenario_id, subject, difficulty, trial_mode,
mode, condition, seed,
success, completion_time_sec,
total_inputs, interactions, motion_tool_calls,
mode_switches, direction_reversals,
dropped_inputs, selection_perturbations, direction_perturbations,
target_filtered_out, terminated_at_max_ticks, end_reason
```

`by_condition.csv` (aggregated — one row per (mode, condition, difficulty)):

```
mode, condition, difficulty, n_rollouts, success_rate,
mean_completion_time_sec, std_completion_time_sec,
mean_total_inputs, std_total_inputs,
mean_interactions, mean_dropped_inputs,
mean_direction_reversals, mean_motion_tool_calls,
target_filtered_out_rate, max_ticks_rate
```

The paper figures join these across (mode × backend × model) and plot
PRIME-vs-Manual deltas at each noise level.

---

## Calibration (§6 of the noise-from-real-data plan)

The scripted user is calibrated **per-scenario** to the priors of the actual
user who ran that trial. At zero noise the sweep's `total_inputs` and
`completion_time_sec` match per-trial PRIME_LOGS `total_commands` and
`total_active_time_sec` within ±25 %:

| Difficulty | sim total_inputs / real total_commands | sim time / real active_time |
|---|---|---|
| Easy | 0.90 ✓ | 1.14 ✓ |
| Hard | 0.84 ✓ | 1.03 ✓ |

The sim does **not** model inter-burst idle / thinking time, which is fine
because the noise-robustness story reports PRIME-vs-Manual **relative**
deltas — idle time appears equally in both modes and cancels.

---

## Known limitations (note these in the paper)

1. **YCB-only.** The user study covered only reach-to-grasp; we don't have
   real-user data for stacking / pouring. Generalization to those envs is
   future work.
2. **The scripted user emits only motion commands and YES/NO/object-pick
   replies.** It does not model giving-up, asking for help, or other
   non-motor behaviors.
3. **Heuristic baselines dialog-loop under selection noise.** This is
   honest: stateless heuristics are state-blind by design, and the
   noise-robustness story is exactly the regime that exposes this. Report
   their failure rate, don't try to fix it.
4. **Reported noise rates are literature-grounded, not empirically measured
   on motor-impaired users.** The experiment is *bridging evidence*, not a
   substitute for clinical validation.

See `plans/noise-from-real-data.md` for the methodological discussion this
section is built on.
