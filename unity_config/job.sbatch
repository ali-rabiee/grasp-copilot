#!/bin/bash
#SBATCH --job-name=grasp_p1_lora
#SBATCH --partition=gpu,uri-gpu
#SBATCH --gpus=1
#SBATCH --constraint=l40s|a100|h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --open-mode=append

# Phase-1 oracle LoRA training for plans/training_woz_envs.md.
#
# Default submit:
#   cd grasp-copilot
#   sbatch unity_config/job.sbatch
#
# Override paths or hyperparameters at submit time, for example:
#   TRAIN_PATH=data/merged_3env_v1/llm_contract_3env_rebalanced.jsonl \
#   OUTPUT_DIR=models/qwen2_5_3b_oracle_lora \
#   sbatch unity_config/job.sbatch
#
# GPU choice:
# - l40s|a100|h100 asks for any of these fast-enough GPUs. This keeps queue
#   flexibility while allowing H100 if one is available.
# - If the queue is blocked and you are willing to use slower cards, change the
#   constraint to vram32.
# - If you want to avoid H100 scarcity entirely, change it back to l40s|a100.

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-copilot}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
TRAIN_PATH="${TRAIN_PATH:-data/merged_3env_v1/llm_contract_3env_rebalanced.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-models/qwen2_5_3b_oracle_lora}"

LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1024}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LR="${LR:-2e-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
LOGGING_STEPS="${LOGGING_STEPS:-25}"
SEED="${SEED:-0}"

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"

if [ ! -f "scripts/train_sft_lora.py" ]; then
    echo "Run this job from the grasp-copilot repo root."
    echo "Current directory: $REPO"
    exit 2
fi

if [ ! -f "$TRAIN_PATH" ]; then
    echo "Training file not found: $TRAIN_PATH"
    echo "Build it first from the data prep section of plans/training_woz_envs.md."
    exit 2
fi

mkdir -p "logs/slurm" "$OUTPUT_DIR"
exec >>"logs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1

echo "[job] started at $(date --iso-8601=seconds)"
echo "[job] node=${SLURM_NODELIST:-unknown} job_id=${SLURM_JOB_ID:-unknown}"
echo "[job] repo=$REPO"
echo "[job] train_path=$TRAIN_PATH"
echo "[job] output_dir=$OUTPUT_DIR"

if [ -f /modules/apps/conda/main/etc/profile.d/conda.sh ]; then
    source /modules/apps/conda/main/etc/profile.d/conda.sh
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

python scripts/train_sft_lora.py \
    --model_name "$MODEL_NAME" \
    --train_path "$TRAIN_PATH" \
    --no-full_finetune --no-use_4bit \
    --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --lr "$LR" --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --warmup_ratio "$WARMUP_RATIO" --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
    --disable_eval \
    --save_strategy no \
    --logging_steps "$LOGGING_STEPS" \
    --merge_at_end \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR"

echo "[job] finished at $(date --iso-8601=seconds)"
