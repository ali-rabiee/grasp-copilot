#!/bin/bash
#SBATCH --job-name=grasp_p2_woz
#SBATCH --partition=gpu,uri-gpu
#SBATCH --gpus=1
#SBATCH --constraint=l40s|a100|h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --open-mode=append

# Phase-2 WOZ LoRA training: adapt the local oracle-trained 3B model on
# data/woz_* generator logs.
#
# Default submit:
#   cd grasp-copilot
#   sbatch unity_config/job.sbatch
#
# Override paths or hyperparameters at submit time, for example:
#   MODEL_NAME=models/qwen2_5_3b_oracle_lora \
#   WOZ_OUT_DIR=data/woz_phase2 \
#   OUTPUT_DIR=models/qwen2_5_3b_oracle_woz_lora \
#   sbatch unity_config/job.sbatch
#
# GPU choice:
# - l40s|a100|h100 asks for any of these fast-enough GPUs. This keeps queue
#   flexibility while allowing H100 if one is available.
# - If the queue is blocked and you are willing to use slower cards, change the
#   constraint to vram32.
# - If you want to avoid H100 scarcity entirely, change it back to l40s|a100.

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llm}"
MODEL_NAME="${MODEL_NAME:-models/qwen2_5_3b_instruct_ft}"
WOZ_OUT_DIR="${WOZ_OUT_DIR:-data/woz_phase2}"
TRAIN_PATH="${TRAIN_PATH:-$WOZ_OUT_DIR/llm_contract_train_rebalanced.jsonl}"
VALID_PATH="${VALID_PATH:-$WOZ_OUT_DIR/llm_contract_valid.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-models/qwen2_5_3b_oracle_woz_lora}"

LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1024}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LR="${LR:-1e-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
LOGGING_STEPS="${LOGGING_STEPS:-25}"
EVAL_STEPS="${EVAL_STEPS:-50}"
SEED="${SEED:-0}"
WOZ_VALID_FRACTION="${WOZ_VALID_FRACTION:-0.10}"
WOZ_MOTION_REPEAT="${WOZ_MOTION_REPEAT:-2}"
WOZ_INTERACT_KEEP_PROB="${WOZ_INTERACT_KEEP_PROB:-1.0}"

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"

if [ ! -f "scripts/train_sft_lora.py" ]; then
    echo "Run this job from the grasp-copilot repo root."
    echo "Current directory: $REPO"
    exit 2
fi

for gen in \
    data/woz_reach_to_grasp_ycb/grasp_gen.jsonl \
    data/woz_cube_stacking/grasp_gen.jsonl \
    data/woz_pouring/grasp_gen.jsonl
do
    if [ ! -f "$gen" ]; then
        echo "WOZ generator file not found: $gen"
        exit 2
    fi
done

if [ -d "$MODEL_NAME" ] && [ ! -f "$MODEL_NAME/config.json" ]; then
    echo "Local model directory is missing config.json: $MODEL_NAME"
    exit 2
fi
if [ -d "$MODEL_NAME" ] && ! compgen -G "$MODEL_NAME/model*.safetensors" >/dev/null && [ ! -f "$MODEL_NAME/pytorch_model.bin" ]; then
    echo "Local model directory is missing model weights: $MODEL_NAME"
    exit 2
fi

if [ ! -f "$TRAIN_PATH" ]; then
    echo "Training file not found yet; it will be prepared at: $TRAIN_PATH"
fi

JOB_NAME="${SLURM_JOB_NAME:-grasp_p2_woz}"
JOB_ID="${SLURM_JOB_ID:-manual}"

mkdir -p "logs/slurm" "$OUTPUT_DIR" "$WOZ_OUT_DIR"
exec >>"logs/slurm/${JOB_NAME}_${JOB_ID}.log" 2>&1

echo "[job] started at $(date --iso-8601=seconds)"
echo "[job] node=${SLURM_NODELIST:-unknown} job_id=$JOB_ID"
echo "[job] repo=$REPO"
echo "[job] model_name=$MODEL_NAME"
echo "[job] woz_out_dir=$WOZ_OUT_DIR"
echo "[job] train_path=$TRAIN_PATH"
echo "[job] valid_path=$VALID_PATH"
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

python -m llm.prepare_woz_phase2_data \
    --out_dir "$WOZ_OUT_DIR" \
    --valid_fraction "$WOZ_VALID_FRACTION" \
    --motion_repeat "$WOZ_MOTION_REPEAT" \
    --interact_keep_prob "$WOZ_INTERACT_KEEP_PROB" \
    --seed "$SEED"

python scripts/train_sft_lora.py \
    --model_name "$MODEL_NAME" \
    --train_path "$TRAIN_PATH" \
    --valid_path "$VALID_PATH" \
    --no-use_4bit \
    --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --lr "$LR" --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --warmup_ratio "$WARMUP_RATIO" --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
    --eval_steps "$EVAL_STEPS" \
    --save_strategy no \
    --logging_steps "$LOGGING_STEPS" \
    --merge_at_end \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR"

echo "[job] finished at $(date --iso-8601=seconds)"
