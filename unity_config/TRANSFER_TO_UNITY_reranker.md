# Transferring the IG-reranker ablation to Unity

Single-page checklist for the Package 4 (IG reranker) sweeps. Read once
before submitting jobs. Mirrors the structure of
`TRANSFER_TO_UNITY_noise_sweep.md` so the conventions match.

---

## 1. What goes through git vs. what's transferred separately

**Goes through git** (commit on dev machine, `git pull` on Unity):

| Path | Size | Why |
|---|---|---|
| `llm/reranker/` (5 modules + tests) | ~25 KB | Post-LLM filter (pruning, entropy, candidates, selector, wrapper) |
| `evaluation/reranker/` (5 modules) | ~30 KB | Sweep shim + dialog logger + analyze + plot + tables |
| `unity_config/job_reranker_ablation.sbatch` | ~7 KB | Online sweep submission |
| `unity_config/job_ig_analysis.sbatch` | ~2 KB | CPU-only post-hoc analysis |
| `unity_config/check_reranker.sh` | ~8 KB | Smoke validator |
| `unity_config/TRANSFER_TO_UNITY_reranker.md` | this file | |
| `plans/package4_ig_reranker.md` | ~25 KB | Authoritative plan |
| Reuse: `evaluation/benchmarks/scenario_noise_sweep.py`, `evaluation/rollouts/`, `evaluation/scenarios/` | unchanged | The reranker imports these — don't re-transfer |
| Reuse: `evaluation/results/robustness/user_input_noise/scenarios/scenarios.labeled.jsonl` | already on Unity | The 160-scenario corpus from Package 3 |

**Transferred separately** (NOT in git — too large):

| Item | Size | How to transfer |
|---|---|---|
| `models/qwen2_5_3b_oracle_woz_lora/` | ~6 GB | likely already on Unity from the noise sweep — verify before transferring |
| `models/qwen2_5_3b_oracle_lora/` | ~6 GB | likewise |
| Conda env `copilot` | — | Already exists on Unity |

No new model artifacts beyond what the user-input-noise sweep already
needed.

---

## 2. One-time setup on Unity

```bash
ssh unity
cd ~/grasp-copilot
git pull

# Sanity-check the conda env can import the new modules:
conda activate copilot
python -c "
from llm.reranker import make_reranked_backend, entropy_bits
from evaluation.reranker.run_reranker_sweep import main
print('reranker imports OK')
"
```

If that import succeeds, no new deps are needed (the reranker reuses
`torch`, `transformers`, `peft` already pinned for the noise sweep).

Verify the scenario corpus and models are present:

```bash
ls -lh evaluation/results/robustness/user_input_noise/scenarios/scenarios.labeled.jsonl
ls -lh models/qwen2_5_3b_oracle_woz_lora/model*.safetensors
ls -lh models/qwen2_5_3b_oracle_lora/model*.safetensors
```

If any are missing, follow `unity_config/TRANSFER_TO_UNITY_noise_sweep.md`
§3 to rsync them.

---

## 3. The four sbatch submissions (one smoke + three headline)

### 3a. Smoke first (≤ 1 h, single GPU)

Validates wiring end-to-end before committing the 20 h budget.

```bash
cd ~/grasp-copilot
N_SEEDS=1 \
CONDITIONS=clean \
MAX_SCENARIOS=5 \
MODEL_KEY=oracle_woz_lora \
MODEL_PATH=models/qwen2_5_3b_oracle_woz_lora \
RERANK_MODE=info_gain \
K_CANDIDATES=5 \
sbatch --time=01:00:00 unity_config/job_reranker_ablation.sbatch
```

Wait for it to finish (~20–40 min), then:

```bash
bash unity_config/check_reranker.sh \
    evaluation/results/reranker/ablation/oracle_woz_lora__info_gain
```

Expect all six checks to pass except possibly check [3] (max-ticks rate
may be elevated on a single-seed clean-condition slice — fine for smoke).

**If anything looks wrong, stop here.** The full sweep is ~50 GPU-h
across three jobs and you don't want to discover a misconfig 18 h in.

### 3b. Headline run #1 — `oracle_woz_lora` × `info_gain` (~18 h GPU)

```bash
MODEL_KEY=oracle_woz_lora \
MODEL_PATH=models/qwen2_5_3b_oracle_woz_lora \
RERANK_MODE=info_gain \
K_CANDIDATES=5 \
RERANK_TEMPERATURE=0.7 \
sbatch unity_config/job_reranker_ablation.sbatch
```

This is the headline cell — produces the "WoZ + IG-rerank" row of
`table_reranker_ablation.tex`.

### 3c. Headline run #2 — `oracle_woz_lora` × `none` (~14 h GPU)

```bash
MODEL_KEY=oracle_woz_lora \
MODEL_PATH=models/qwen2_5_3b_oracle_woz_lora \
RERANK_MODE=none \
sbatch unity_config/job_reranker_ablation.sbatch
```

`RERANK_MODE=none` skips candidate generation but still logs every
emitted INTERACT to `dialogs.jsonl`, so the offline analysis can replay
the same dialogs through `random` and `oracle` selectors for free.

### 3d. Headline run #3 — `oracle_lora` × `info_gain` (~18 h GPU)

```bash
MODEL_KEY=oracle_lora \
MODEL_PATH=models/qwen2_5_3b_oracle_lora \
RERANK_MODE=info_gain \
K_CANDIDATES=5 \
sbatch unity_config/job_reranker_ablation.sbatch
```

Sanity-check row: shows whether the reranker rescues the warm-start
checkpoint to the level of the WoZ-trained model, or whether WoZ adds
something the reranker can't recover.

**These three jobs run in parallel** if three GPU nodes are free. With
queue contention, expect ~24–48 h end-to-end wallclock.

### 3e. (Optional) `RERANK_MODE=random` as a within-policy control

Per the plan §1.4, the random/oracle selector comparisons come for free
from offline replay of the `RERANK_MODE=none` log — you do **not** need
to run a separate `RERANK_MODE=random` sweep. Skip unless you want a
true online random control.

---

## 3f. Contract IG analysis (CPU-only, runs anywhere)

Sensitivity check for the appendix: measure IG of every INTERACT in the
held-out contract eval sets (`data/woz_phase2/llm_contract_valid.jsonl`
plus the three `data/oracle_valid_*` sets). No GPU, no model load, no
sweep — just reads the contracts and applies the same pruning + entropy
modules. Finishes in ~30 seconds.

Run on **dev or Unity**, your choice. On dev:

```bash
cd /home/ali/github/ali-rabiee/grasp-copilot
conda activate llm    # or `copilot` on Unity
python -m evaluation.reranker.analyze_ig_contracts
# Outputs land in evaluation/results/reranker/ig_analysis_contracts/
```

Produces:
```
evaluation/results/reranker/ig_analysis_contracts/
├── per_question_contracts.csv
├── summary_contracts.json
├── appendix_ig_contracts.csv
└── appendix_ig_contracts.tex     ← appendix table for the paper
```

Reading the table: oracle contracts contain many structural INTERACT
calls (intent_gate, mode_select, anything_else) that don't shrink the
candidate set — IG on them is 0 by construction. So this table reports
a **lower bound** on PRIME's question informativeness; the
scenario-sweep IG in the main body is the headline number.

---

## 4. After all three sbatch finish: the analysis pass

CPU-only, ~5 min. Run on Unity (or pull results to dev and run locally —
either works, the analysis is small).

```bash
sbatch unity_config/job_ig_analysis.sbatch
```

This produces:

```
evaluation/results/reranker/
├── ig_analysis/
│   ├── per_question.csv          # all dialogs × 4 selectors
│   ├── summary.json              # aggregate IG per selector + by kind
│   ├── ig_distribution.pdf       # paper figure: histogram per selector
│   ├── ig_distribution.png
│   ├── ig_by_kind.pdf            # facet: QUESTION / CONFIRM / SUGGESTION
│   └── ig_by_kind.png
└── tables/
    ├── table_reranker_ablation.{csv,tex}   # headline ablation table
    └── table_ig_summary.{csv,tex}          # per-selector IG summary
```

The two `.tex` files use booktabs and are ready to paste into
`paper_snippets.tex` next to the existing PRIME tables.

---

## 5. Where outputs land on Unity

```
evaluation/results/reranker/
├── ablation/
│   ├── oracle_woz_lora__info_gain/
│   │   ├── rollouts.csv          (~9,600 rows per cell)
│   │   ├── by_condition.csv
│   │   ├── dialogs.jsonl         (per-INTERACT IG records)
│   │   └── sweep_meta.json
│   ├── oracle_woz_lora__none/
│   │   └── (same files)
│   └── oracle_lora__info_gain/
│       └── (same files)
├── ig_analysis/                   # produced by job_ig_analysis.sbatch
└── tables/                        # likewise
```

Per-job slurm logs land in `logs/slurm/rerank_sweep_<cell>__<job_id>.log`.

---

## 6. Pulling results back to dev machine

```bash
rsync -avh --progress \
    unity:~/grasp-copilot/evaluation/results/reranker/ \
    /home/ali/github/ali-rabiee/grasp-copilot/evaluation/results/reranker/
```

Per the repo `.gitignore`, only `by_condition.csv`, `sweep_meta.json`,
`summary.json`, and the tables/figures get committed back.
`rollouts.csv` and `dialogs.jsonl` stay local on whichever machine last
produced them — both are recomputable from a re-submission.

---

## 7. Monitoring

While jobs are running:

```bash
ssh unity
squeue -u $USER                            # are my jobs running?
tail -f ~/grasp-copilot/logs/slurm/rerank_sweep_*.log   # live progress
wc -l ~/grasp-copilot/evaluation/results/reranker/ablation/oracle_woz_lora__info_gain/rollouts.csv
                                            # rollouts completed
```

The sweep prints a progress line every 100 rollouts with a live ETA.
`dialogs.jsonl` grows continuously — `wc -l` on it tells you how many
INTERACT decisions have been logged so far.

---

## 8. Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `Model dir not found: models/...` | Model not transferred or wrong path | rsync per §1; verify with `ls ~/grasp-copilot/models/<name>/` |
| `Scenarios file not found` | `git pull` missed the noise-sweep commits | `git pull` then re-submit |
| `AttributeError: 'NoneType' object has no attribute 'model_path'` | Old llm/reranker checkout | `git pull`; this was fixed when candidates.py learned to skip sampling when no LLM is loaded |
| Sweep takes >20 h and SLURM cancels | K=5 + full 6-condition matrix on a slow GPU | Drop K to 3 (cuts ~40% time) or run with `CONDITIONS="clean compound_mid"` first |
| `CUDA out of memory` | Concurrent LLM in cache | Each sbatch loads one model; submit one (model, rerank_mode) per job — don't combine |
| `dialogs.jsonl` empty after smoke | RERANK_MODE=none with logging disabled on a non-prime mode | Confirm `MODES=prime` and `BACKEND=hf_ft`; the smoke template above sets both |
| `check_reranker.sh` step [5] fails | IG values out of bounds → bug in pruning.py | Open one failing record from dialogs.jsonl; verify `n_candidates_before > 0` and `ig_bits <= log2(n_candidates_before)` |
| `check_reranker.sh` step [3] flags low success on smoke | Expected with N_SEEDS=1 and a 5-scenario slice | Ignore for smoke; the headline runs have N_SEEDS=5 |

---

## 9. End-to-end checklist (copy-paste sequence)

```bash
# 0. ON DEV: commit + push the package4 code
cd /home/ali/github/ali-rabiee/grasp-copilot
git add llm/reranker/ evaluation/reranker/ unity_config/job_reranker_ablation.sbatch \
        unity_config/job_ig_analysis.sbatch unity_config/check_reranker.sh \
        unity_config/TRANSFER_TO_UNITY_reranker.md plans/package4_ig_reranker.md
git commit -m "package 4: information-gain reranker + sbatch templates"
git push

# 1. ON UNITY: pull and verify imports
ssh unity
cd ~/grasp-copilot
git pull
conda activate copilot
python -c "from llm.reranker import make_reranked_backend; print('OK')"

# 2. SMOKE first
N_SEEDS=1 CONDITIONS=clean MAX_SCENARIOS=5 \
MODEL_KEY=oracle_woz_lora MODEL_PATH=models/qwen2_5_3b_oracle_woz_lora \
RERANK_MODE=info_gain \
sbatch --time=01:00:00 unity_config/job_reranker_ablation.sbatch
# wait for completion, then:
bash unity_config/check_reranker.sh \
    evaluation/results/reranker/ablation/oracle_woz_lora__info_gain

# 3. HEADLINE — three submissions, runnable in parallel
MODEL_KEY=oracle_woz_lora MODEL_PATH=models/qwen2_5_3b_oracle_woz_lora \
    RERANK_MODE=info_gain sbatch unity_config/job_reranker_ablation.sbatch
MODEL_KEY=oracle_woz_lora MODEL_PATH=models/qwen2_5_3b_oracle_woz_lora \
    RERANK_MODE=none      sbatch unity_config/job_reranker_ablation.sbatch
MODEL_KEY=oracle_lora     MODEL_PATH=models/qwen2_5_3b_oracle_lora \
    RERANK_MODE=info_gain sbatch unity_config/job_reranker_ablation.sbatch

# 4. AFTER all three complete: build tables + figures
sbatch unity_config/job_ig_analysis.sbatch

# 4b. (anywhere, anytime) Contract IG appendix table — CPU only, ~30 s
python -m evaluation.reranker.analyze_ig_contracts

# 5. PULL results back to dev
exit  # leave Unity
rsync -avh --progress \
    unity:~/grasp-copilot/evaluation/results/reranker/ \
    /home/ali/github/ali-rabiee/grasp-copilot/evaluation/results/reranker/
```

That sequence is the entire Package 4 ablation. The two `.tex` files
under `evaluation/results/reranker/tables/` are the artifacts that drop
into the paper.

---

## 10. What gets read back into the paper

| Output | Goes into | Paper claim it backs |
|---|---|---|
| `tables/table_reranker_ablation.tex` | Section IV-E, Table 3 (new) | "WoZ + reranker vs WoZ alone vs Oracle + reranker" |
| `tables/table_ig_summary.tex` | Section IV-E, Table 4 (new) | "Per-selector mean IG, fraction ≥ 0.5 bits" |
| `ig_analysis/ig_distribution.pdf` | Section IV-E, Fig 8 (new) | "PRIME's questions deliver X bits IG on average" |
| `ig_analysis/ig_by_kind.pdf` | Appendix figure | "IG broken down by INTERACT kind" |
| `ig_analysis_contracts/appendix_ig_contracts.tex` | Appendix table | "Sensitivity: IG holds across held-out contract eval sets" |
| `dialogs.jsonl` (any cell) | Not in paper directly | Source of truth — keep for reviewer requests |

If `mean_ig_chosen ≥ 0.5 bits`, the validation gate from the plan §11
is met. If it isn't, **stop and investigate before writing the paragraph
in Section IV-E** — the WoZ model is producing low-signal questions and
the headline claim doesn't hold.
