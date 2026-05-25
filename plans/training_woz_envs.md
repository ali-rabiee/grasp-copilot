# Training Plan: Wizard-of-Oz Models Across Three Environments

**Project:** PRIME IROS 2026 extension — replace the heuristic-oracle teacher with a Wizard-of-Oz (WoZ) teacher across YCB reach-to-grasp, cube stacking, and pouring environments.

**Status:** Re-drafted 2026-05-13 for Qwen2.5-3B on a single V100 32GB. Authoritative until the first row of Panel A is filled in with real numbers.

**Owner:** Ali Rabiee (lead). Co-authors: MH Farhadi, Shayan Khodabakhsh, Resit Sendag, Reza Abiri.

---

## 0 · TL;DR

- **One unified Qwen2.5-3B model** handles all three environments. Per-env scores come from a single evaluation, not three separate trainings.
- **Two-phase LoRA**: Phase 1 LoRA on oracle data warm-starts the schema; Phase 2 LoRA on WoZ data refines judgement. Full-FT is dropped — V100 + 3B + 8h budget makes LoRA the only reliable path, and Biderman et al. 2024 shows LoRA matches full FT at our data scale.
- **Headline model**: `qwen2_5_3b_oracle_woz_lora`. Headline baseline: `qwen2_5_3b_oracle_lora` (Phase-1 checkpoint alone, no WoZ).
- **5 training runs total, ~15 GPU-hours on a single V100 32GB.** Every run targets <8h wallclock.
- **WoZ data target: 800 episodes per env (~8k decisions/env, ~24k total)**, collected with `wizard/` GUI by ≥2 wizards. Pilot at 300 ep/env first.
- **Critical second table**: "Oracle vs Oracle→WoZ on ambiguous-scenario subset" — this is where WoZ has to win clearly, or the paper story collapses.
- **No mid-training eval.** In-training valid loss was burning ~80% of wallclock in the old recipe. Hold-out evaluation happens once, after training, via `evaluation/run_full_benchmark.py`.

---

## 1 · Strategic decisions and their justification

### 1.1 Per-env vs unified → **unified single model**

Train one 3B that handles all three environments, evaluated on per-env held-out sets. The PRIME paper already advertises "modular and extensible" architecture; per-env models contradict that framing and force a defensive paragraph.

Literature anchors:

- **Padalkar et al., 2024 — Open X-Embodiment / RT-X** [^padalkar2024]. Single VLA across 22 embodiments, 21 institutions. Shows positive transfer across very heterogeneous tasks.
- **Kim et al., 2024 — OpenVLA** [^kim2024openvla]. One model across many manipulation tasks; the canonical recent example.
- **Brohan et al., 2023 — RT-2** [^brohan2023rt2]. Vision-language-action; single multitask model.
- **Huang et al., 2022 — Inner Monologue** [^huang2022inner]. One LLM, multiple skills.

Sanity-check ablation row to include in the appendix: "per-env separate models" — train three 3Bs separately, evaluate on the same held-out, show the unified model isn't sacrificing per-env accuracy. If a per-env model wins by >1 pp on any env, dig in.

### 1.2 Model size → **Qwen2.5-3B-Instruct**

Switched from the original 7B target because the available training hardware is a single V100 32GB. Three justifications beyond hardware:

1. **Deployment story.** PRIME runs closed-loop at ~5 Hz; a 3B with quantization is far closer to deployable than a 7B. The paper's narrative already favors the smaller model.
2. **Baseline parity.** The existing PRIME paper's numbers in `qwen2_5_3b_instruct_ft` already use 3B, so the comparison is apples-to-apples.
3. **LIMA-style argument.** [^zhou2023lima] At our SFT scale, the bottleneck is data quality (the WoZ argument), not capacity. Going from 3B → 7B at <30k examples buys little.

If a 7B sweep becomes feasible later (A100/H100 access), re-running Models #3 and #4 at 7B is a clean appendix add — the LoRA recipe transfers without changes.

### 1.3 Supervision strategy → **Oracle LoRA warm-start, then WoZ LoRA refinement**

Three options were considered:

| Option | Pros | Cons |
|---|---|---|
| Pure WoZ (LoRA on base Qwen) | Cleanest "WoZ replaces oracle" story | Risks under-training; wastes the oracle's free structural signal |
| Mixed data (oracle + WoZ in one run) | What RT-2 / OpenVLA do | Confounds: "is the win from WoZ or just more data?" |
| **Oracle LoRA → WoZ LoRA (sequential, two adapters)** ✓ | Clean attribution: oracle gives structure, WoZ gives judgement. Sample-efficient. V100-friendly. | Two-step recipe to explain |

Story we get to claim: *"the oracle teaches the schema (JSON shape, tool vocabulary, prompt types); WoZ teaches the policy under ambiguity (which oracle thresholds get wrong)."* This maps cleanly to the paper's existing "oracle is a training-time teacher, not deployable" framing.

Phase 2 uses the **merged Phase-1 checkpoint** as its base model (the Phase-1 LoRA adapter is merged into the base weights before Phase 2 begins), so Phase 2 trains a fresh LoRA adapter on top of an oracle-tuned dense model. This avoids stacking adapters at inference time and keeps the final deployable artifact a single merged 3B.

Literature anchors for continued / staged fine-tuning:

- **Biderman et al., 2024 — "LoRA Learns Less and Forgets Less"** [^biderman2024lora]. LoRA preserves base capabilities; ideal as a refinement step on top of an oracle-tuned checkpoint.
- **Tülu / Tülu 2 (Wang et al., 2023; Ivison et al., 2023)** [^wang2023tulu] [^ivison2023tulu2]. SFT plateaus around 10–50k examples; staged training keeps each phase well-conditioned.
- **OpenVLA fine-tuning sections** [^kim2024openvla]. Pre-train on robotics-broad data, fine-tune per-task — directly analogous.

### 1.4 Data sizing → **800 episodes per env, ~8k decisions/env, ~24k total**

Per-env plan:

| Phase | Episodes | Decisions (≈10/ep) | Purpose |
|---|---|---|---|
| **Pilot** | 300 / env (900 total) | 3k / env (9k total) | Validate κ ≥ 0.7 inter-wizard agreement, confirm training picks up the WoZ signal vs oracle baseline. |
| **Production** | 800 / env (2400 total) | 8k / env (24k total) | Headline numbers. |
| **Held-out agreement subset** | 50 / env (150 total) | ~500 / env | Annotated by *all* wizards independently; never enters training. |

Wizard-hours estimate: ~30 s/decision in GUI × 24k decisions = ~200 wizard-hours. With 3 wizards that's ~70 hours each.

**Oracle data: 5k episodes per env after rebalancing → ~30k rows total → ~20k rows in train (after carving valid).** Down from 10k/env in the previous plan; at the LoRA scale this is enough, and it keeps Phase 1 well under 8h on V100.

---

## 2 · Models to train

Five training runs. Each corresponds to a row in the headline table.

| # | Model name | Base | Supervision | Method | Role |
|---|---|---|---|---|---|
| 1 | `qwen2_5_3b_oracle_lora` | Qwen2.5-3B-Instruct | Oracle (3 envs, ~20k rows) | LoRA r=16 α=32 | Phase-1 checkpoint; supervision-source baseline. |
| 2 | `qwen2_5_3b_woz_lora` | Qwen2.5-3B-Instruct | WoZ only (~22k rows) | LoRA r=16 α=32 | "Pure WoZ" — shows oracle warm-start helps. |
| 3 | **`qwen2_5_3b_oracle_woz_lora`** | #1 merged checkpoint | WoZ on top | LoRA r=16 α=32 | **Headline model**. |
| 4 | `qwen2_5_3b_oracle_woz_lora_r32` | #1 merged checkpoint | WoZ on top | LoRA r=32 α=64 | LoRA-rank ablation (replaces the 7B variant from the old plan; opportunistic if time allows). |
| 5 | `qwen2_5_3b_oracle_lora_perEnv` | Qwen2.5-3B-Instruct | Oracle per-env (3 separate models) | LoRA r=16 α=32 | Per-env vs unified sanity ablation. |

Plus zero-shot 3B numbers (no training) and heuristic baselines (H1/H2/SA1/SA2 — also no training).

If V100 time opens up to A100/H100 later, add `qwen2_5_7b_oracle_woz_lora` (Model #6) as an appendix entry — same recipe, just swap the base model.

---

## 3 · Exact training recipes

### V100 + LoRA conventions (apply to every recipe below)

These settings are non-negotiable on V100; deviate only if you've benchmarked the change yourself.

- **Precision: fp16, not bf16.** V100 has no bf16 hardware support; the training script auto-detects and falls back to fp16 + GradScaler.
- **No mid-training eval.** Pass `--disable_eval`. The valid file built in §3.0 exists for *post-hoc* spot checks via `evaluation/run_full_benchmark.py --contract_jsonl …`, not as an in-training callback. The previous plan's recipe burned ~80% of wallclock on eval passes.
- **`max_seq_length=1024`.** Verified on `data/ycb/01/llm_contract_1000.jsonl`: median 1849 chars, p95 2646 chars → ~460 / ~660 tokens. 1024 covers >99% of examples; 2048 is wasted padding.
- **`per_device_train_batch_size=4`, `gradient_accumulation_steps=4`** → effective batch 16. On V100 32GB at seq_len 1024 with LoRA + gradient checkpointing this leaves ~10 GB free.
- **Gradient checkpointing ON** (the script forces it). Costs ~20% step time but is what makes batch 4 fit.
- **Save final adapter only.** `--save_strategy no` plus the script's automatic end-of-run `save_pretrained`. Mid-run checkpoints buy nothing if you're not going to resume.
- **fp16-safe optimizer**: `--optim adamw_torch` (default). Do not use `paged_adamw_32bit` here — that's a QLoRA-only optimization.
- **16GB V100 fallback**: drop `--no-use_4bit` from every command below to enable QLoRA. Step time roughly doubles; everything else identical.

### 3.0 Data prep (run once)

```bash
cd grasp-copilot
conda activate llm

# Collect oracle data (5k episodes per env is enough for LoRA Phase 1).
grasp-collect --env reach_to_grasp_ycb --episodes 5000 --seed 0 --rebalance
grasp-collect --env cube_stacking      --episodes 5000 --seed 0 --rebalance
grasp-collect --env pouring            --episodes 5000 --seed 0 --rebalance

mkdir -p data/merged_3env_v1

# Build the rebalanced training pool (shuffled with deterministic seed).
cat data/ycb/01/llm_contract_5000_rebalanced.jsonl \
    data/stacking/01/llm_contract_5000_rebalanced.jsonl \
    data/pouring/01/llm_contract_5000_rebalanced.jsonl \
  | shuf --random-source=<(yes 42) \
  > data/merged_3env_v1/_pool_oracle.jsonl

# Hold out 1000 lines for occasional post-hoc loss spot-checks (not used during training).
head -n 1000 data/merged_3env_v1/_pool_oracle.jsonl \
    > data/merged_3env_v1/llm_contract_3env_valid_small.jsonl
tail -n +1001 data/merged_3env_v1/_pool_oracle.jsonl \
    > data/merged_3env_v1/llm_contract_3env_rebalanced.jsonl

# Sanity-check disjointness by `id`.
python -c "
import json
trn = {json.loads(l)['id'] for l in open('data/merged_3env_v1/llm_contract_3env_rebalanced.jsonl')}
val = {json.loads(l)['id'] for l in open('data/merged_3env_v1/llm_contract_3env_valid_small.jsonl')}
assert (trn & val) == set(), 'valid leaked into train'
print(f'train={len(trn)} valid={len(val)} disjoint=ok')
"

rm data/merged_3env_v1/_pool_oracle.jsonl
```

Expected sizes after this: ~19–21k train rows, 1000 valid rows. The exact number depends on rebalancing.

### 3.1 Phase 1 — Oracle LoRA (Model #1)

```bash
python scripts/train_sft_lora.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --train_path data/merged_3env_v1/llm_contract_3env_rebalanced.jsonl \
    --no-full_finetune --no-use_4bit \
    --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
    --max_seq_length 1024 \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
    --lr 2e-4 --num_train_epochs 1 \
    --warmup_ratio 0.03 --lr_scheduler_type cosine \
    --disable_eval \
    --save_strategy no \
    --logging_steps 25 \
    --merge_at_end \
    --output_dir models/qwen2_5_3b_oracle_lora
```


Effective batch = 16. Step count ≈ 20k / 16 × 2 epochs ≈ 2500 optimizer steps. On V100 32GB at ~6 s/step → **~4 h wallclock**.

`--merge_at_end` writes the adapter into a standalone merged model at `models/qwen2_5_3b_oracle_lora/`; Phase 2 points at this directory.

### 3.2 Phase 2 — WoZ LoRA on top (Model #3, headline)

Current Unity workflow: prepare `data/woz_phase2` explicitly first, then submit
`unity_config/job.sbatch`. The Slurm job expects prepared contract JSONL files
and does not create them.

Prepare the current WOZ generator folders:

```bash
cd ~/ali/grasp-copilot
conda activate copilot
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python -m llm.prepare_woz_phase2_data \
    --out_dir data/woz_phase2 \
    --valid_fraction 0.10 \
    --motion_repeat 2 \
    --interact_keep_prob 1.0 \
    --seed 0
```

Expected output for the current 100-episode-per-env WOZ folders:

```json
{
  "rows": {
    "all": 1500,
    "train": 1350,
    "train_rebalanced": 1707,
    "valid": 150
  },
  "tool_distribution": {
    "train_rebalanced": {
      "ALIGN_YAW": 90,
      "APPROACH": 166,
      "GRAB": 180,
      "INTERACT": 993,
      "POUR": 90,
      "STACK": 188
    }
  }
}
```

Verify the files exist:

```bash
ls -lh data/woz_phase2
cat data/woz_phase2/summary.json
```

Required files for the Slurm job:

```text
data/woz_phase2/llm_contract_train_rebalanced.jsonl
data/woz_phase2/llm_contract_valid.jsonl
```

Submit the Phase-2 job:

```bash
sbatch unity_config/job.sbatch
```

The current job defaults are:

```text
MODEL_NAME=models/qwen2_5_3b_instruct_ft
TRAIN_PATH=data/woz_phase2/llm_contract_train_rebalanced.jsonl
VALID_PATH=data/woz_phase2/llm_contract_valid.jsonl
OUTPUT_DIR=models/qwen2_5_3b_oracle_woz_lora
```

If `models/qwen2_5_3b_oracle_lora` is repaired and complete, use it as the
Phase-2 base:

```bash
MODEL_NAME=models/qwen2_5_3b_oracle_lora sbatch unity_config/job.sbatch
```

Monitor paths:

```bash
scontrol show job <JOB_ID> | grep -E 'WorkDir=|StdOut=|StdErr='
tail -f logs/slurm/grasp_p2_woz_<JOB_ID>.log
```

### 3.3 Model #2 — Pure WoZ LoRA (no oracle warm-start)

Identical to §3.2 except `--model_name Qwen/Qwen2.5-3B-Instruct` (base, not the Phase-1 merged checkpoint). Used only to argue oracle warm-start helps (claim #3 in §6).

### 3.4 Model #4 — LoRA rank ablation (optional)

Identical to §3.2 except `--lora_r 32 --lora_alpha 64`. Same step count, ~10% more wallclock from the larger adapter. Run only if Models #1–#3 finish under budget.

### 3.5 Model #5 — Per-env LoRAs (ablation)

Three runs, identical to §3.1 but pointed at the per-env unmerged rebalanced files:

```bash
for env in ycb stacking pouring; do
    python scripts/train_sft_lora.py \
        --model_name Qwen/Qwen2.5-3B-Instruct \
        --train_path data/$env/01/llm_contract_5000_rebalanced.jsonl \
        --no-full_finetune --no-use_4bit \
        --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
        --max_seq_length 1024 \
        --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
        --lr 2e-4 --num_train_epochs 2 \
        --warmup_ratio 0.03 --lr_scheduler_type cosine \
        --disable_eval --save_strategy no --logging_steps 25 --merge_at_end \
        --output_dir models/qwen2_5_3b_oracle_lora_${env}
done
```

Each run ≈ 1.5h on V100 32GB. Total ~4.5h for all three.

---

## 4 · Evaluation plan

Re-use the existing offline benchmark machinery in `grasp-copilot/evaluation/`.

### 4.1 Eval sets per env

These are the authoritative paper-table eval sets — separate from the small in-training valid files (§3.0) which are post-hoc spot-check loss only.

| Eval set | Size | Source | Use |
|---|---|---|---|
| `oracle_eval_ycb` | 1,947 (existing) | Oracle generator, held-out | Direct comparison to existing Table I numbers. |
| `oracle_eval_stacking` | ~2,000 | Oracle generator, fresh | Per-env tool accuracy / strict-match for new envs. |
| `oracle_eval_pouring` | ~2,000 | Oracle generator, fresh | Per-env tool accuracy / strict-match for new envs. |
| `ambiguous_eval` | 200–300 curated | Hand-picked from the held-out agreement subset + curated additions | The critical "WoZ beats oracle" eval. |
| `robustness_user_noise` | Same 1947 with p∈{0.1, 0.2, 0.3, 0.5} | Perturb gripper history | Matches current Table II protocol. |

### 4.2 Metrics (unchanged from current Table I)

- Tool accuracy
- Strict exact match (full tool + args, excluding INTERACT text)
- Motion-object accuracy
- Interact-kind accuracy
- Schema validity (JSON, args shape)
- Throughput (ex/s)

Run via:

```bash
python -m evaluation.run_full_benchmark \
    --model_dir models/qwen2_5_3b_oracle_woz_lora \
    --eval_sets oracle_eval_ycb,oracle_eval_stacking,oracle_eval_pouring \
    --out_dir evaluation/eval_outputs/paper_benchmark_v2
```

Plus a new `evaluation/ambiguous_scenarios_benchmark.py` script for the WoZ-specific table (write this).

---

## 5 · Paper tables and figures

### 5.1 Table I — Headline (replaces current Table I)

**Panel A: main models**

| Model | Supervision | Method | YCB | Stack | Pour | All-3 avg | Schema valid | Trainable params | ex/s |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B | ZS | — | 16.6 | ? | ? | ? | 89.6 | 0 | 3.01 |
| Qwen2.5-3B | Oracle | LoRA | ? | ? | ? | ? | ~100 | ~25M | ~1.4 |
| Qwen2.5-3B | WoZ | LoRA | ? | ? | ? | ? | ~100 | ~25M | ~1.4 |
| **Qwen2.5-3B** | **Oracle→WoZ** | **LoRA** | **?** | **?** | **?** | **?** | **~100** | **~25M** | **~1.4** |

**Panel B: baselines (per-env tool accuracy)**

| Method | YCB | Stack | Pour | All-3 avg |
|---|---|---|---|---|
| Heuristic oracle (teacher) | 96.1 | ? | ? | ? |
| H1: Ask-if-Ambiguous | 92.9 | ? | ? | ? |
| H2: Always-Ask | 94.8 | ? | ? | ? |
| SA1: Predict-then-Assist | 81.1 | ? | ? | ? |
| SA2: Bayesian Intent | 85.4 | ? | ? | ? |

### 5.2 Table II — WoZ-specific contribution (NEW; this is the WoZ paper's central claim)

| | Tool acc on **ambiguous scenarios** | Tool acc on **clean scenarios** |
|---|---|---|
| Heuristic oracle (teacher) | ? | 96.1 |
| Qwen2.5-3B-Oracle-LoRA | ? | ? |
| **Qwen2.5-3B-Oracle→WoZ-LoRA** | **?** | **?** |

Build the ambiguous-scenarios set as soon as the WoZ pilot data is in. It's the load-bearing element of the paper.

### 5.3 Table III — Robustness (replaces current Table II)

Same as the existing robustness table, with the WoZ headline model added.

### 5.4 Ablation appendix table

| Ablation | Cells / variants |
|---|---|
| Per-env vs unified | Model #5 (3 per-env LoRAs) vs Model #3 (unified Oracle→WoZ-LoRA) |
| LoRA rank | r=16 (Model #3) vs r=32 (Model #4) |
| Warm-start | Oracle→WoZ (#3) vs pure-WoZ (#2) |
| Data scale | WoZ episodes per env ∈ {300, 500, 800} (drop the 1500 cell to stay in budget) |

### 5.5 Figures

Existing figures to update:

| File | Update |
|---|---|
| `fig1_radar.png` | Add Oracle→WoZ-LoRA series (3B). |
| `fig5_confusion_grid.png` | Add new tool types (STACK, GRAB, POUR). |
| User study composites (`user_study_res1/res2.png`) | Re-collect user study on Oracle→WoZ-LoRA. |

New figures to author:

| Figure | Content |
|---|---|
| **WoZ data-scale curve** | Tool accuracy vs # WoZ episodes per env (300, 500, 800). Two lines: clean eval, ambiguous eval. The ambiguous line is the load-bearing one. |
| **Inter-wizard κ bar chart** | From the agreement subset; reports Cohen's κ (pairwise) and Fleiss' κ (3+ wizards). Goal κ ≥ 0.7. |
| **Tradeoff plot** | Trainable params (x, log) vs accuracy (y). Shows base LoRA vs higher-rank LoRA; main model in the sweet spot. |

---

## 6 · Putting it together — the paper claim chain

In the order the paper must argue them:

1. **WoZ ≥ Oracle on clean scenarios** — Panel A: Oracle→WoZ-LoRA matches Oracle-LoRA on every env. (If this fails, the paper story collapses; abort and investigate before continuing.)
2. **WoZ > Oracle on ambiguous scenarios** — Table II. The central WoZ contribution.
3. **Warm-start helps** — Model #3 > Model #2 in Panel A.
4. **LoRA at r=16 is sufficient** — Model #3 ≈ Model #4 (rank 32), so we don't need more capacity in the adapter.
5. **One unified model handles all three envs** — Panel A per-env columns + Model #5 ablation.
6. **Inter-wizard κ ≥ 0.7** — supporting result, preempts "your wizards are inconsistent."

---

## 7 · Compute budget (V100 32GB, single GPU)

| Phase | Wallclock |
|---|---|
| Oracle data generation (15k episodes total) | ~30 min CPU |
| WoZ data collection | ~200 wizard-hours (humans, not GPU) |
| Model #1 (Oracle LoRA) | ~4 h |
| Model #3 (Oracle→WoZ LoRA, headline) | ~4–5 h |
| Model #2 (Pure WoZ LoRA) | ~4–5 h |
| Model #4 (LoRA rank 32, optional) | ~5 h |
| Model #5 (Per-env LoRAs ×3) | ~4–5 h total |
| Evaluation runs (~5 models × benchmark) | ~3–5 h |
| **Total GPU time** | **~25–30 h** spread across the 4 weeks; no single run exceeds 8 h. |

All training commands target <8h on V100 32GB. If any run exceeds 6h, halve the epoch count first (LoRA on this data scale typically converges in 1 epoch — 2 is insurance).

---

## 8 · Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| WoZ matches but doesn't *beat* the oracle on ambiguous scenarios | Medium | Pilot at 300 ep/env before committing to 800. If pilot is flat, redesign the ambiguous-scenarios set (curate harder cases) before scaling up. |
| Inter-wizard κ < 0.7 | Medium | Add a wizard-training session before data collection: walk all wizards through ~20 example states with discussion. |
| LoRA r=16 underperforms — model under-capacity | Low | Model #4 (r=32) is the planned response. If r=32 wins by >2pp, promote it to headline. |
| Unified model is worse than per-env on any env | Low | Model #5 already planned. If unified loses on one env, frame the unified model as "trades 1 pp for one-deployment simplicity"; per-env stays in appendix. |
| Wizard hours blow up | Medium | The 300-ep pilot is the failsafe: stop at pilot if κ is below threshold or training signal is unclear. |
| V100 step time worse than estimated (>10 s/step) | Medium | Drop epochs from 2 → 1 (typically loses <0.5 pp). Then drop `lora_target_modules` to attention-only (`q,k,v,o`) — saves ~30% step time. Last resort: switch to QLoRA (`--use_4bit`), accepting ~10–20% step time penalty but lower memory pressure. |
| V100 OOM at seq_len 1024 | Low | Drop micro-batch to 2 (eff batch 8, doubles steps but halves memory); or seq_len 768 (covers p90 of contracts). |
| fp16 loss instability (NaN losses) | Low | The script handles GradScaler. If it triggers, lower `--lr` to 1e-4 and re-run. Schema-validity ≥99% gate (run on the 1k valid file after training) catches silent breakage. |

---

## 9 · Suggested timeline

A focused 4-week sprint. Assumes wizards are available and committed.

| Week | Tasks |
|---|---|
| **Week 1** | Oracle data regeneration (3 envs × 5k ep). Train Model #1 (Phase-1 3B-Oracle-LoRA, ~4 h). Build `ambiguous_eval` curated set (~200 examples). Validate Model #1 on per-env benchmarks → fills the "Oracle LoRA" row of Panel A. |
| **Week 2** | Pilot WoZ collection (300 ep × 3 envs × 2 wizards). Compute pilot inter-wizard κ. Train a pilot Model #3 on the 300-ep data (~3 h). Verify it isn't worse than oracle on clean eval and shows *some* signal on ambiguous eval. |
| **Week 3** | Scale WoZ collection to 800 ep / env. Train Models #2 and #3 (~10 h total V100 time). Run benchmark on both. |
| **Week 4** | Train Models #4 and #5 (rank + per-env ablations, ~10 h V100 time). Build figures. Write Section IV (Experiments) updates. Re-run robustness sweep on Model #3. |

---

## 10 · Citations

Full bibliographic details for the literature anchors used above. All keys match the citation keys in `P_Rabiee_Copilot_IROS_2026/root.bib`; verify before final submission.

[^hu2021lora]: Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). **LoRA: Low-Rank Adaptation of Large Language Models.** arXiv:2106.09685. Defines LoRA; r=8–32 with α=2r is the common default range.

[^dettmers2023qlora]: Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). **QLoRA: Efficient Finetuning of Quantized LLMs.** NeurIPS 2023. 4-bit base + LoRA; viable on V100 16GB.

[^biderman2024lora]: Biderman, D., Ortiz, J. G., Portes, J., Paul, M., Greengard, P., Jennings, C., King, D., Havens, S., Chiley, V., Frankle, J., Blakeney, C., & Cunningham, J. P. (2024). **LoRA Learns Less and Forgets Less.** TMLR / NeurIPS 2024. The canonical recent comparison; LoRA matches full FT on instruction-tuning at our data scale. The "forgets less" half is the warm-start argument.

[^zhou2023lima]: Zhou, C., Liu, P., Xu, P., Iyer, S., Sun, J., Mao, Y., Ma, X., Efrat, A., Yu, P., Yu, L., Zhang, S., Ghosh, G., Lewis, M., Zettlemoyer, L., & Levy, O. (2023). **LIMA: Less Is More for Alignment.** NeurIPS 2023. 1,000 curated examples ≈ 100k+ noisy examples for SFT.

[^wang2023tulu]: Wang, Y., Ivison, H., Dasigi, P., Hessel, J., Khot, T., Chandu, K. R., Wadden, D., MacMillan, K., Smith, N. A., Beltagy, I., & Hajishirzi, H. (2023). **How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources (Tülu).** NeurIPS 2023.

[^ivison2023tulu2]: Ivison, H., Wang, Y., Pyatkin, V., Lambert, N., Peters, M., Dasigi, P., Jang, J., Wadden, D., Smith, N. A., Beltagy, I., & Hajishirzi, H. (2023). **Camels in a Changing Climate: Enhancing LM Adaptation with Tülu 2.** arXiv:2311.10702.

[^padalkar2024]: Open X-Embodiment Collaboration: Padalkar, A., et al. (2024). **Open X-Embodiment: Robotic Learning Datasets and RT-X Models.** ICRA 2024.

[^kim2024openvla]: Kim, M. J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E., Lam, G., Sanketi, P., Vuong, Q., Kollar, T., Burchfiel, B., Tedrake, R., Sadigh, D., Levine, S., Liang, P., & Finn, C. (2024). **OpenVLA: An Open-Source Vision-Language-Action Model.** Cited in PRIME as `kim2024openvla`.

[^brohan2023rt2]: Brohan, A., Brown, N., Carbajal, J., et al. (2023). **RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control.** CoRL 2023. Cited as `zitkovich2023rt2` in PRIME's bib.

[^huang2022inner]: Huang, W., Xia, F., Xiao, T., Chan, H., Liang, J., Florence, P., Zeng, A., Tompson, J., Mordatch, I., Chebotar, Y., Sermanet, P., Brown, N., Jackson, T., Luu, L., Levine, S., Hausman, K., & Ichter, B. (2022). **Inner Monologue: Embodied Reasoning through Planning with Language Models.** CoRL 2022. Cited in PRIME as `huang2022inner`.

[^ahn2022saycan]: Ahn, M., Brohan, A., Brown, N., et al. (2022). **Do As I Can, Not As I Say: Grounding Language in Robotic Affordances (SayCan).** CoRL 2022. Cited in PRIME as `ahn2022can`.

**Still need to be added to the .bib** for this plan:
- `hu2021lora`
- `dettmers2023qlora`
- `biderman2024lora`
- `zhou2023lima`
- `wang2023tulu`
- `padalkar2024openx`

---

## 11 · Key files and pointers

- **Oracle code**: `grasp-copilot/data_generator/oracle.py`, `oracle_stacking.py`, `oracle_pouring.py`, `oracle_registry.py`.
- **Episode samplers**: `grasp-copilot/data_generator/episode.py`, `episode_stacking.py`, `episode_pouring.py`.
- **WoZ collection**: `grasp-copilot/wizard/` (entry: `python -m wizard collect …`).
- **Visual oracle/model tester**: `grasp-copilot/scripts/gui_assist_demo.py` (supports `--env` and `--backend {oracle, hf}`).
- **Training entry**: `grasp-copilot/scripts/train_sft_lora.py` (delegates to `llm/train.py`).
- **Evaluation**: `grasp-copilot/evaluation/run_full_benchmark.py`.
- **Existing oracle-FT checkpoints**: `grasp-copilot/models/qwen2_5_3b_instruct_ft` (YCB-only; superseded by `qwen2_5_3b_oracle_lora`).
- **Existing eval outputs**: `grasp-copilot/evaluation/eval_outputs/paper_benchmark_run001/` (YCB Table I numbers); `robustness_user_input_1947/` (Table II numbers).
- **Paper source**: `P_Rabiee_Copilot_IROS_2026/root.tex`, `root.bib`, figures in `figs/`.
- **User study analysis**: `user-study-prime/` (re-run on the new headline model in Week 4).

---

## 12 · Open questions for follow-up sessions

These are deliberate punts — decide them when the data lands:

1. **Ambiguous-scenarios set composition**: Hand-curated only, or hand-curated + programmatically-mined (e.g., scenes where the oracle's `_has_cell_oscillation` returns True)? Mining is cheaper but might miss the human-intuitive ambiguities.
2. **WoZ episode mix per wizard**: Each wizard does all 3 envs, or partition (one wizard per env)? Partition is faster but loses inter-wizard κ per env. Default: all wizards do all envs.
3. **Do we re-run the user study on the WoZ model?** Strongly recommended. Could rerun with the same N=8 participants if available, or recruit a fresh pool.
4. **7B appendix run**: If A100/H100 access opens, run `qwen2_5_7b_oracle_woz_lora` with the same recipe and add as Model #6 in an appendix table. Decide based on Model #4 (rank 32) results — if r=32 buys >1 pp on ambiguous eval, 7B is worth chasing.
