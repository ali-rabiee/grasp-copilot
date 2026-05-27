#!/bin/bash
#SBATCH --job-name=rerank_sweep
#SBATCH --partition=gpu,uri-gpu
#SBATCH --gpus=1
#SBATCH --constraint=l40s|a100|h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --open-mode=append

# IG-reranker ablation sweep — one (model, rerank_mode) per submission.
# (see plans/package4_ig_reranker.md §9.1).
#
# Wraps the existing scenario-noise sweep with the reranker plus a
# dialogs.jsonl log that captures every emitted INTERACT for the offline
# IG analysis. Each submission runs the full (141 x 6 x 1 x N_SEEDS)
# matrix for ONE model x rerank_mode pair; submit several jobs to fill
# the headline ablation table in plans/package4_ig_reranker.md §7.
#
# Examples (from grasp-copilot repo root):
#
#   # Headline:
#   MODEL_KEY=oracle_woz_v2 \
#   MODEL_PATH=models/v2/qwen2_5_3b_oracle_woz_v2 \
#       RERANK_MODE=info_gain \
#       sbatch unity_config/job_reranker_ablation.sbatch
#
#   # Baseline (no reranker, but still logs dialogs for offline replay):
#   MODEL_KEY=oracle_woz_v2 \
#   MODEL_PATH=models/v2/qwen2_5_3b_oracle_woz_v2 \
#       RERANK_MODE=none \
#       sbatch unity_config/job_reranker_ablation.sbatch
#
#   # Warm-start sanity check:
#   MODEL_KEY=oracle_lora MODEL_PATH=models/qwen2_5_3b_oracle_lora \
#       RERANK_MODE=info_gain \
#       sbatch unity_config/job_reranker_ablation.sbatch
#
# Tunable env vars (with defaults):
#   BACKEND         hf_ft | oracle | heuristic           (default: hf_ft)
#   MODEL_KEY       short name for hf_ft sweep           (required when BACKEND=hf_ft)
#   MODEL_PATH      path to fine-tuned model             (required when BACKEND=hf_ft)
#   HEURISTIC       one of HEURISTIC_REGISTRY            (required when BACKEND=heuristic)
#   RERANK_MODE     none | info_gain | random | oracle   (default: info_gain)
#   K_CANDIDATES    number of candidate samples          (default: 5)
#   RERANK_TEMPERATURE  sampling temperature             (default: 0.7)
#   RERANK_TOP_P    nucleus sampling top-p               (default: 0.95)
#   PRIOR           uniform | motion_weighted            (default: uniform)
#   MODES           subset of "manual prime"             (default: "prime")
#   N_SEEDS         rollouts per (scenario, mode, cond)  (default: 5)
#   MAX_TICKS       hard cap on rollout length           (default: 500)
#   CONDITIONS      space-separated noise conditions     (default: all 6)
#   MAX_SCENARIOS   0 = all scenarios                    (default: 0)
#   CONDA_ENV       conda env name                       (default: copilot)
#
# Time budget: ~20 h for the full hf_ft matrix at RERANK_MODE=info_gain
# (5x LLM cost on INTERACT ticks). RERANK_MODE=none mirrors the noise
# sweep's ~14 h. Drop CONDITIONS or N_SEEDS to shorten.

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-copilot}"
BACKEND="${BACKEND:-hf_ft}"
MODEL_KEY="${MODEL_KEY:-}"
MODEL_PATH="${MODEL_PATH:-}"
HEURISTIC="${HEURISTIC:-}"
RERANK_MODE="${RERANK_MODE:-info_gain}"
K_CANDIDATES="${K_CANDIDATES:-5}"
RERANK_TEMPERATURE="${RERANK_TEMPERATURE:-0.7}"
RERANK_TOP_P="${RERANK_TOP_P:-0.95}"
PRIOR="${PRIOR:-uniform}"
MODES="${MODES:-prime}"
N_SEEDS="${N_SEEDS:-5}"
MAX_TICKS="${MAX_TICKS:-500}"
CONDITIONS="${CONDITIONS:-clean dir_low dir_high sel_low sel_high compound_mid}"
MAX_SCENARIOS="${MAX_SCENARIOS:-0}"
RERANK_SEED="${RERANK_SEED:-0}"

SCENARIOS="evaluation/results/robustness/user_input_noise/scenarios/scenarios.labeled.jsonl"

if [ "$BACKEND" = "hf_ft" ]; then
    [ -n "$MODEL_KEY"  ] || { echo "MODEL_KEY required for BACKEND=hf_ft";  exit 2; }
    [ -n "$MODEL_PATH" ] || { echo "MODEL_PATH required for BACKEND=hf_ft"; exit 2; }
    SWEEP_NAME="${MODEL_KEY}__${RERANK_MODE}"
    EXTRA_ARGS=(--backend hf_ft --model_paths "$MODEL_KEY=$MODEL_PATH")
elif [ "$BACKEND" = "heuristic" ]; then
    [ -n "$HEURISTIC" ] || { echo "HEURISTIC required for BACKEND=heuristic"; exit 2; }
    SWEEP_NAME="${HEURISTIC}__${RERANK_MODE}"
    EXTRA_ARGS=(--backend heuristic --heuristic "$HEURISTIC")
else
    SWEEP_NAME="${BACKEND}__${RERANK_MODE}"
    EXTRA_ARGS=(--backend "$BACKEND")
fi

OUT_DIR="evaluation/results/reranker/ablation/$SWEEP_NAME"

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"

if [ ! -f "$SCENARIOS" ]; then
    echo "Scenarios file not found: $SCENARIOS"
    echo "Run extract_from_prime_logs first; see"
    echo "  evaluation/results/robustness/user_input_noise/README.md"
    exit 2
fi

if [ "$BACKEND" = "hf_ft" ]; then
    if [ ! -d "$MODEL_PATH" ]; then
        echo "Model dir not found: $MODEL_PATH"
        echo "Transfer the merged model tree from your dev machine to Unity first."
        exit 2
    fi
    if ! compgen -G "$MODEL_PATH/model*.safetensors" >/dev/null && [ ! -f "$MODEL_PATH/pytorch_model.bin" ]; then
        echo "Model dir missing weights: $MODEL_PATH"
        exit 2
    fi
fi

JOB_NAME="${SLURM_JOB_NAME:-rerank_sweep}"
JOB_ID="${SLURM_JOB_ID:-manual}"
mkdir -p "logs/slurm" "$OUT_DIR"
exec >>"logs/slurm/${JOB_NAME}_${SWEEP_NAME}_${JOB_ID}.log" 2>&1

echo "[job] started at $(date --iso-8601=seconds)"
echo "[job] node=${SLURM_NODELIST:-unknown}  job_id=$JOB_ID"
echo "[job] repo=$REPO"
echo "[job] backend=$BACKEND  sweep_name=$SWEEP_NAME"
echo "[job] model_key=${MODEL_KEY:-(n/a)}  model_path=${MODEL_PATH:-(n/a)}"
echo "[job] rerank_mode=$RERANK_MODE  k=$K_CANDIDATES  temp=$RERANK_TEMPERATURE  prior=$PRIOR"
echo "[job] modes=$MODES  n_seeds=$N_SEEDS  conditions=$CONDITIONS"
echo "[job] max_ticks=$MAX_TICKS  max_scenarios=$MAX_SCENARIOS"
echo "[job] out_dir=$OUT_DIR"

if [ -f /modules/apps/conda/main/etc/profile.d/conda.sh ]; then
    source /modules/apps/conda/main/etc/profile.d/conda.sh
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

DIALOG_LOG="$OUT_DIR/dialogs.jsonl"

EXTRA_SCEN=()
if [ "$MAX_SCENARIOS" -gt 0 ]; then
    EXTRA_SCEN=(--max_scenarios "$MAX_SCENARIOS")
fi

python -m evaluation.reranker.run_reranker_sweep \
    --scenarios   "$SCENARIOS" \
    --out_dir     "$OUT_DIR" \
    --modes       $MODES \
    --conditions  $CONDITIONS \
    --n_seeds     "$N_SEEDS" \
    --max_ticks   "$MAX_TICKS" \
    --rerank_mode "$RERANK_MODE" \
    --k_candidates "$K_CANDIDATES" \
    --rerank_temperature "$RERANK_TEMPERATURE" \
    --rerank_top_p "$RERANK_TOP_P" \
    --prior       "$PRIOR" \
    --rerank_seed "$RERANK_SEED" \
    --dialog_log  "$DIALOG_LOG" \
    --progress_every 100 \
    "${EXTRA_SCEN[@]}" \
    "${EXTRA_ARGS[@]}"

echo "[job] finished at $(date --iso-8601=seconds)"
