# PRIME paper benchmark — comprehensive evaluation pipeline

This package evaluates every trained Qwen2.5-3B LoRA against every held-out
dataset (oracle valid × 3 envs, WoZ valid, ambiguous × 3 envs), with four
heuristic / shared-autonomy baselines for reference, and produces conference-
paper-grade LaTeX tables and PDF figures.

---

## TL;DR — one command

```bash
cd /home/ali/github/ali-rabiee/grasp-copilot
conda activate llm

nohup python -m evaluation.run_paper_benchmark > \
    evaluation/eval_outputs/paper_benchmark/logs/run.log 2>&1 &
disown
```

That sweeps **11 models × 7 eval sets = 77 cells**. Re-running the same command
is **safe** — every cell is cached as JSON, so completed cells are skipped on
restart. After it finishes (or any time, on partial caches):

```bash
python -m evaluation.tables.build_paper_tables   # writes LaTeX + CSV
python -m evaluation.plots.paper_figures         # writes PDF + PNG
```

For the noise-robustness sweep (separate run):

```bash
python -m evaluation.run_robustness_sweep --max_examples 300
```

---

## What gets evaluated

### Models (`safe_name → directory`)
| safe_name | display | path |
|---|---|---|
| `oracle_lora` | Qwen2.5-3B-Oracle-LoRA | `models/qwen2_5_3b_oracle_lora` |
| `woz_lora` | Qwen2.5-3B-WoZ-LoRA | `models/qwen2_5_3b_woz_lora` |
| `oracle_woz_lora` | **Qwen2.5-3B-Oracle→WoZ-LoRA** (headline) | `models/qwen2_5_3b_oracle_woz_lora` |
| `oracle_woz_r32` | Oracle→WoZ-LoRA-r32 (rank ablation) | `models/qwen2_5_3b_oracle_woz_lora_r32` |
| `oracle_ycb` | Oracle-LoRA (YCB-only) | `models/qwen2_5_3b_oracle_lora_ycb` |
| `oracle_stacking` | Oracle-LoRA (Stack-only) | `models/qwen2_5_3b_oracle_lora_stacking` |
| `oracle_pouring` | Oracle-LoRA (Pour-only) | `models/qwen2_5_3b_oracle_lora_pouring` |
| `qwen3b_zs` | Qwen2.5-3B-Instruct (zero-shot) | `Qwen/Qwen2.5-3B-Instruct` (HF) — opt-in via `--include_zero_shot` |
| `h1_ask_if_amb` | H1 Ask-if-Ambiguous | rule-based |
| `h2_always_ask` | H2 Always-Ask | rule-based |
| `sa1_pred_assist` | SA1 Predict-then-Assist | shared-autonomy baseline |
| `sa2_bayes_intent` | SA2 Bayesian Intent | shared-autonomy baseline |

### Eval sets (`name → file → size`)
| name | env | flavor | file | n rows |
|---|---|---|---|---|
| `oracle_valid_ycb` | ycb | oracle | `data/oracle_valid_ycb/llm_contract_200.jsonl` | 538 |
| `oracle_valid_stacking` | stacking | oracle | `data/oracle_valid_stacking/llm_contract_200.jsonl` | 937 |
| `oracle_valid_pouring` | pouring | oracle | `data/oracle_valid_pouring/llm_contract_200.jsonl` | 1311 |
| `woz_valid` | mixed | woz | `data/woz_phase2/llm_contract_valid.jsonl` | 150 |
| `ambiguous_ycb` | ycb | ambiguous | `data/_contracts_ambiguous/ambiguous_reach_to_grasp_ycb.jsonl` | 75 |
| `ambiguous_stacking` | stacking | ambiguous | `data/_contracts_ambiguous/ambiguous_cube_stacking.jsonl` | 60 |
| `ambiguous_pouring` | pouring | ambiguous | `data/_contracts_ambiguous/ambiguous_pouring.jsonl` | 75 |

The three `oracle_valid_*` sets were regenerated with `grasp-collect ... --seed 999`
so they're disjoint from training. The three `ambiguous_*` sets are converted
from `data/ambiguous_eval_*/grasp_gen.jsonl` via
`evaluation/convert_ambiguous_to_contract.py`.

---

## Where things land

```
evaluation/eval_outputs/paper_benchmark/
├── logs/
│   ├── run.log                      # main runner stdout/stderr
│   └── heuristics.log               # heuristics-only parallel runs (optional)
├── results/                         # one JSON per (model, eval_set)
│   ├── oracle_lora__oracle_valid_ycb.json
│   ├── oracle_lora__oracle_valid_stacking.json
│   ├── ...
│   └── sa2_bayes_intent__ambiguous_pouring.json
├── mistakes/                        # error-mode JSONLs for analysis
│   └── <model>__<eval_set>.jsonl    # up to 200 mistakes per cell
├── summary_all.csv                  # flat per-cell metrics (overwritten on every run)
├── context_breakdown.csv            # tool-accuracy by dialog context
├── confusion_matrices.csv           # GT × Pred tool confusion matrices
├── manifest.json                    # which models / eval sets / N cells
│
├── tables/                          # output of build_paper_tables (latex + csv)
│   ├── table_1_main.{csv,tex}       # per-env headline
│   ├── table_2_ambiguous.{csv,tex}  # ambiguous vs clean
│   ├── table_3_ablations.{csv,tex}  # warm-start, rank, per-env vs unified
│   └── table_full_metrics.csv       # every metric for every cell
│
├── figures/                         # output of paper_figures (pdf + png)
│   ├── fig1_per_env_bars.{pdf,png}
│   ├── fig2_ambiguous_gap.{pdf,png}
│   ├── fig3_radar.{pdf,png}
│   ├── fig4_confusion_grid.{pdf,png}
│   ├── fig5_context_heatmap.{pdf,png}
│   ├── fig6_accuracy_throughput.{pdf,png}
│   └── fig7_error_breakdown.{pdf,png}
│
└── robustness/                      # output of run_robustness_sweep
    ├── sweep.csv                    # per (model, env, noise_level) row
    ├── sweep_baselines.csv          # Oracle + H1 reference curves
    ├── robustness_curves.{pdf,png}
    └── logs/                        # optional
```

Each per-cell JSON in `results/` carries:
- `tool_accuracy`, `motion_obj_accuracy`, `motion_tool_accuracy`,
  `interact_kind_accuracy`, `interact_choices_valid_rate`,
  `schema_valid_rate`, `strict_exact_rate`
- `tool_confusion` (GT × predicted)
- `by_context`, `by_mode`, `by_num_candidates` breakdowns
- `timing` (`load_s`, `eval_s`, `examples_per_sec`)
- `_model_safe`, `_eval_set`, `_env`, `_flavor`, `_group`, `_display` metadata

The runner is **resume-safe**: cell files are written atomically per cell, and
the runner skips any (model, eval_set) whose JSON already exists.

---

## Running

### 1. The full sweep

```bash
cd /home/ali/github/ali-rabiee/grasp-copilot
conda activate llm
mkdir -p evaluation/eval_outputs/paper_benchmark/logs

nohup python -m evaluation.run_paper_benchmark > \
    evaluation/eval_outputs/paper_benchmark/logs/run.log 2>&1 &
disown
```

**Time estimate (RTX 4080 12 GB):** ~12 min for the smallest eval set, ~25 min
for the largest. Per LoRA across all 7 eval sets: ~90 min. Full sweep of all
7 LoRAs: **~10–11 h**. Heuristics finish in seconds.

**Save power on a laptop.** This is multi-hour GPU compute. Plug in. If the
laptop sleeps or the battery dies, restart with the same command and the
runner will resume from the cache.

**Monitor progress:**

```bash
# Tail the human-readable lines (skip the model-loading progress bars):
grep -aE "^# \[|tool_acc=" \
    evaluation/eval_outputs/paper_benchmark/logs/run.log | tail -20

# Count finished cells (out of 77):
ls evaluation/eval_outputs/paper_benchmark/results/ | wc -l

# Is the process still alive?
pgrep -fa run_paper_benchmark
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
```

### 2. Useful flags

```bash
# Just one model (everything else from cache):
python -m evaluation.run_paper_benchmark --models oracle_woz_lora

# Just one eval set across all models:
python -m evaluation.run_paper_benchmark --eval_sets ambiguous_ycb

# Heuristics only (no GPU, ~30 s):
python -m evaluation.run_paper_benchmark --skip_trained

# Add the zero-shot row (downloads Qwen2.5-3B-Instruct from HF on first run):
python -m evaluation.run_paper_benchmark --include_zero_shot --skip_trained --skip_heuristics

# Quick smoke check (5 examples per cell):
python -m evaluation.run_paper_benchmark --max_examples 5 --rerun

# Force recompute even when cached:
python -m evaluation.run_paper_benchmark --rerun

# 4-bit quantization if VRAM is tight (~50 % memory; ~1.5× slower):
python -m evaluation.run_paper_benchmark --use_4bit

# Faster generation: cap output to 128 tokens (valid tool-call JSONs fit easily).
# ~2× speedup on LoRAs that learned verbose outputs:
python -m evaluation.run_paper_benchmark --max_new_tokens 128
```

### 3. After (or during) the sweep — tables & figures

```bash
# LaTeX + CSV (booktabs style, ready to paste):
python -m evaluation.tables.build_paper_tables
# → evaluation/eval_outputs/paper_benchmark/tables/

# Figures (vector PDF + PNG preview):
python -m evaluation.plots.paper_figures --eval_set oracle_valid_ycb
# → evaluation/eval_outputs/paper_benchmark/figures/
# Re-run with a different --eval_set to redraw the single-set figures
# (radar / confusion / context heatmap / throughput / error breakdown)
# focused on a different env.
```

Both scripts handle partial caches gracefully — missing cells render as `--`
or blank panels, so they're useful for spot-checking mid-run too.

### 4. Robustness sweep (separate run, ~2–3 h GPU)

```bash
# (a) Pre-compute Oracle + H1 reference curves on CPU (instant; safe to run
#     anytime, even while the main benchmark is using the GPU):
python -m evaluation.run_robustness_sweep --baselines_only --max_examples 0

# (b) Full LLM sweep — needs the GPU, so run after run_paper_benchmark is done:
nohup python -m evaluation.run_robustness_sweep --max_examples 300 > \
    evaluation/eval_outputs/paper_benchmark/robustness/logs/sweep.log 2>&1 &
disown
```

This sweeps **7 LoRAs × 3 envs × 5 noise levels** of the `user_input`
perturbation (`p ∈ {0, 0.1, 0.2, 0.3, 0.5}`). Per-cell rows are appended to
`robustness/sweep.csv`; restarting the command skips already-recorded cells.
Oracle + H1 baseline curves live in a separate `sweep_baselines.csv`. The plot
lands in `robustness/robustness_curves.pdf`.

To only plot from existing data (works with baselines alone, or baselines +
LLM sweep):

```bash
python -m evaluation.run_robustness_sweep --plot_only
```

### 4a. One-shot post-benchmark finisher

After the main benchmark finishes, this script does the remaining work in
order: optional zero-shot row → LLM robustness sweep → tables + figures.

```bash
bash evaluation/finish_paper_run.sh             # full (~3 h GPU)
bash evaluation/finish_paper_run.sh --skip-zs   # skip the 7 GB HF download
bash evaluation/finish_paper_run.sh --tables-only   # just rebuild outputs
```

### 5. End-to-end one-shot (after a fresh data prep)

```bash
# Convert ambiguous-eval grasp_gen → contract format (only needed once):
python -m evaluation.convert_ambiguous_to_contract

# Regenerate per-env oracle valid sets with held-out seed (only needed once):
grasp-collect --env reach_to_grasp_ycb --episodes 200 --seed 999 --out_dir data/oracle_valid_ycb
grasp-collect --env cube_stacking      --episodes 200 --seed 999 --out_dir data/oracle_valid_stacking
grasp-collect --env pouring            --episodes 200 --seed 999 --out_dir data/oracle_valid_pouring

# Then the full pipeline:
nohup python -m evaluation.run_paper_benchmark > \
    evaluation/eval_outputs/paper_benchmark/logs/run.log 2>&1 &
disown

python -m evaluation.tables.build_paper_tables
python -m evaluation.plots.paper_figures
```

---

## Tables produced

| File | Content |
|---|---|
| `table_1_main.tex` | Per-env tool accuracy (YCB / Stack / Pour / Avg-3 / WoZ) for trained LoRAs and baselines. |
| `table_2_ambiguous.tex` | Clean Avg-3 vs Ambiguous Avg-3 per model with Δpp. This is the central WoZ claim. |
| `table_3_ablations.tex` | Three panels: warm-start (`woz_lora` vs `oracle_lora` vs `oracle_woz_lora`); LoRA rank (r=16 vs r=32); per-env vs unified. |
| `table_full_metrics.csv` | Every metric for every (model, eval_set) — machine-readable. |

All `.tex` files are `\begin{table}` … `\end{table}` blocks using `booktabs`
(`\toprule` / `\midrule` / `\bottomrule`) with `\textbf{}` on best-per-column.
Drop them into the paper source as-is.

## Figures produced

| File | What it shows |
|---|---|
| `fig1_per_env_bars.pdf` | Grouped bars: tool accuracy per env, grouped by model. |
| `fig2_ambiguous_gap.pdf` | Paired clean-vs-ambiguous bars with gap-arrows. |
| `fig3_radar.pdf` | Radar of 5 metrics (tool / motion-obj / interact-kind / schema / strict). |
| `fig4_confusion_grid.pdf` | Row-normalized tool confusion matrices, 2×3 grid. |
| `fig5_context_heatmap.pdf` | Tool accuracy by dialog-context type × model. |
| `fig6_accuracy_throughput.pdf` | ex/s vs tool accuracy scatter (log x-axis). |
| `fig7_error_breakdown.pdf` | Stacked horizontal bar: correct / wrong tool / schema invalid / JSON invalid. |
| `robustness/robustness_curves.pdf` | Tool accuracy vs noise level, one subplot per env, one line per LoRA + dashed Oracle/H1 references. |

All figures are produced as both `.pdf` (vector, paper) and `.png` (200 DPI
preview). Matplotlib settings embed Type-42 fonts so the PDFs are TrueType-
editable in the final paper.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Process dies silently mid-run (laptop sleep / battery / SIGTERM) | Just rerun the same command. Cells already finished are cached and will be skipped. |
| `INVALID_JSON` or `INVALID_SCHEMA` in confusion matrix | Inspect `mistakes/<model>__<eval_set>.jsonl` — the raw model output is dumped there. |
| `Some parameters are on the meta device because they were offloaded to the cpu.` | GPU memory pressure. Either close other GPU users, or rerun with `--use_4bit`. |
| Tables / figures show `--` or empty panels | Those (model, eval_set) cells haven't been computed yet. Run the benchmark for them. |
| Need to re-evaluate a single bad cell | `python -m evaluation.run_paper_benchmark --models <safe_name> --eval_sets <eval_name> --rerun` |
| Want only the headline numbers, fast | `--max_examples 200 --rerun` — gives noisier but quickly-iterating numbers. |

---

## Files in this package

| File | Purpose |
|---|---|
| `convert_ambiguous_to_contract.py` | Convert `data/ambiguous_eval_*/grasp_gen.jsonl` → contract format. |
| `offline_exec_benchmark.py` | Core scoring + heuristic baselines (callable as a single-shot CLI too). |
| `run_paper_benchmark.py` | **Top-level driver.** Loads each model once and sweeps it across all eval sets, caching per-cell. |
| `run_robustness_sweep.py` | Noise-level sweep across all LoRAs × envs. |
| `robustness_benchmark.py` | Underlying perturbation registry + Oracle/H1 scoring utilities. |
| `tables/build_paper_tables.py` | Produces `table_{1,2,3}.{csv,tex}` and `table_full_metrics.csv`. |
| `plots/paper_figures.py` | Produces `fig{1..7}.{pdf,png}`. |
| `plots/make_offline_exec_figures.py` | Older single-eval-set figure script (preserved). |
| `scenarios/` | Scenario-seeded simulation utilities (Package 3 / noise-from-real-data plan). |
