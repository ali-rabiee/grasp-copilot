# Transferring the user-input-noise sweep to Unity

Single-page checklist for getting the noise sweep running on Unity. Read this
once before submitting jobs.

---

## 1. What goes through git vs. what's transferred separately

**Goes through git** (commit on dev machine, `git pull` on Unity):

| Path | Size | Why |
|---|---|---|
| `evaluation/benchmarks/scenario_noise_sweep.py` | ~25 KB | Sweep runner |
| `evaluation/rollouts/` | ~30 KB | Noise channels, scripted user, rollout loop |
| `evaluation/scenarios/` | ~50 KB | Scenario extraction + adapters (already committed) |
| `evaluation/results/robustness/README.md` + subdir READMEs | ~15 KB | Documentation |
| `evaluation/results/robustness/user_input_noise/scenarios/*.jsonl` | ~600 KB | **The 160-scenario corpus + labels — committed** |
| `evaluation/results/robustness/user_input_noise/scenarios/scenarios_summary.json` | ~8 KB | Provenance metadata |
| `evaluation/results/robustness/perception_noise/sweep*.csv` + figures | ~200 KB | Existing perception-noise results |
| `unity_config/job_noise_sweep.sbatch` | ~6 KB | This SBATCH file |
| `unity_config/TRANSFER_TO_UNITY_noise_sweep.md` | this file | |
| `data_generator/episode.py`, `data_generator/oracle.py` | unchanged | Sim engine the sweep imports |

**Transferred separately** (NOT in git — too large or environment-specific):

| Item | Size | How to transfer |
|---|---|---|
| `models/qwen2_5_3b_oracle_woz_lora/` | ~6 GB | `rsync -avh --progress models/qwen2_5_3b_oracle_woz_lora/ unity:~/grasp-copilot/models/qwen2_5_3b_oracle_woz_lora/` |
| `models/qwen2_5_3b_oracle_lora/` | ~6 GB | same pattern |
| `models/qwen2_5_3b_woz_lora/` | ~6 GB | same pattern (only if running the pure-WoZ ablation) |
| Conda env `copilot` | — | Already exists on Unity (used by `unity_config/job.sbatch`) |
| `PRIME_LOGS/` | — | Not needed on Unity — scenarios are already extracted and committed |

Total model transfer: ~18 GB for all 3 LLMs. The heuristic and manual
backends don't need models at all.

---

## 2. One-time setup on Unity

```bash
ssh unity
cd ~/grasp-copilot                  # or wherever you keep the repo
git pull
mkdir -p models                     # if it doesn't already exist
```

Confirm the conda env has everything (matches the existing
`unity_config/job.sbatch` requirements — no new deps for the sweep):

```bash
conda activate copilot
python -c "import torch, transformers, peft, llm.inference; print('OK')"
```

---

## 3. Transfer the models (from dev machine)

Run these on your dev machine. `rsync` handles resume on disconnect:

```bash
# Headline:
rsync -avh --progress \
    /home/ali/github/ali-rabiee/grasp-copilot/models/qwen2_5_3b_oracle_woz_lora/ \
    unity:~/grasp-copilot/models/qwen2_5_3b_oracle_woz_lora/

# Warm-start ablation:
rsync -avh --progress \
    /home/ali/github/ali-rabiee/grasp-copilot/models/qwen2_5_3b_oracle_lora/ \
    unity:~/grasp-copilot/models/qwen2_5_3b_oracle_lora/

# Pure-WoZ control (optional — drops the third LLM from the plan if you skip it):
rsync -avh --progress \
    /home/ali/github/ali-rabiee/grasp-copilot/models/qwen2_5_3b_woz_lora/ \
    unity:~/grasp-copilot/models/qwen2_5_3b_woz_lora/
```

Verify each model's weight files made it:

```bash
ssh unity 'ls -lh ~/grasp-copilot/models/qwen2_5_3b_oracle_woz_lora/model-*.safetensors'
```

---

## 4. Submit the jobs

The sweep submits **one job per model**, plus optional heuristic baselines.
Run from the repo root on Unity (`cd ~/grasp-copilot`).

### Headline + warm-start ablation (the main paper claim)

```bash
# Job 1: headline
MODEL_KEY=oracle_woz_lora \
MODEL_PATH=models/qwen2_5_3b_oracle_woz_lora \
sbatch unity_config/job_noise_sweep.sbatch

# Job 2: warm-start ablation
MODEL_KEY=oracle_lora \
MODEL_PATH=models/qwen2_5_3b_oracle_lora \
sbatch unity_config/job_noise_sweep.sbatch
```

These run in parallel if two GPUs are free. Each takes ~20 h wall-clock for
the full 9,600-rollout matrix.

### Pure-WoZ control (optional third LLM)

```bash
MODEL_KEY=woz_lora \
MODEL_PATH=models/qwen2_5_3b_woz_lora \
sbatch unity_config/job_noise_sweep.sbatch
```

### Heuristic + manual baselines (cheap, run while LLM jobs are queued)

```bash
# Stateless heuristics — each takes <1 h:
for h in h1_ask_if_amb h2_always_ask sa1_pred_assist sa2_bayes_intent; do
    BACKEND=heuristic HEURISTIC=$h \
    sbatch unity_config/job_noise_sweep.sbatch
done

# Manual baseline (CPU only — finishes in minutes):
BACKEND=manual MODES=manual \
sbatch unity_config/job_noise_sweep.sbatch
```

### Smoke test before the long runs

```bash
N_SEEDS=1 \
MODEL_KEY=oracle_woz_lora \
MODEL_PATH=models/qwen2_5_3b_oracle_woz_lora \
CONDITIONS=clean \
sbatch --time=01:00:00 unity_config/job_noise_sweep.sbatch
```

This should finish in <30 minutes and confirm the model loads + first
batch of rollouts complete cleanly. Check the output before launching the
20-hour runs.

---

## 5. Where the outputs land on Unity

```
evaluation/results/robustness/user_input_noise/sweeps/
├── oracle_woz_lora/
│   ├── rollouts.csv          (per-rollout, ~9,600 rows)
│   ├── by_condition.csv      (aggregated)
│   └── sweep_meta.json       (run config + timing)
├── oracle_lora/
├── woz_lora/                 (if you ran the third LLM)
├── h1_ask_if_amb/
├── h2_always_ask/
├── sa1_pred_assist/
├── sa2_bayes_intent/
└── manual/
```

Per-job slurm logs land in `logs/slurm/noise_sweep_<sweep_name>_<job_id>.log`.

---

## 6. Pulling results back to dev machine

```bash
rsync -avh --progress \
    unity:~/grasp-copilot/evaluation/results/robustness/user_input_noise/sweeps/ \
    /home/ali/github/ali-rabiee/grasp-copilot/evaluation/results/robustness/user_input_noise/sweeps/
```

Per the repo `.gitignore`, only `by_condition.csv` and `sweep_meta.json`
end up committed back. `rollouts.csv` stays local on whichever machine
last produced it (recomputable from the sweep if needed).

---

## 7. Monitoring

While a job is running:

```bash
ssh unity
squeue -u $USER                                          # are my jobs running?
tail -f ~/grasp-copilot/logs/slurm/noise_sweep_*.log     # progress
wc -l ~/grasp-copilot/evaluation/results/robustness/user_input_noise/sweeps/oracle_woz_lora/rollouts.csv
                                                          # rollouts completed so far
```

The sweep prints a progress line every 100 rollouts including a live ETA.

---

## 8. Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `Model dir not found: models/...` | Model not transferred or wrong path | `rsync` it; verify with `ls ~/grasp-copilot/models/<name>/` |
| `Scenarios file not found` | Forgot to `git pull` after the scenario commit | `git pull` |
| Slurm cancels job at 20:00:00 | Run is genuinely longer than budget | Resubmit with fewer scenarios (`--max_scenarios 50`) or fewer seeds |
| `CUDA out of memory` | Multiple models in cache (shouldn't happen with this script — bug) | Confirm `_free_hf_cache()` is being called between models; submit one model per job |
| Heuristic sweep shows 0 % success | Expected — stateless heuristics dialog-loop. See `user_input_noise/README.md` "Known limitations" |
