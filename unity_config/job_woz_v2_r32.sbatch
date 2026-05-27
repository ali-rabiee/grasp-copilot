#!/bin/bash
#SBATCH --job-name=grasp_combo_r32
#SBATCH --partition=gpu,uri-gpu
#SBATCH --gpus=1
#SBATCH --constraint=l40s|a100|h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --open-mode=append

# Independent all-env Oracle+WoZ v2-data rank ablation.
# Trains from base Qwen with the same v2 data as the headline
# v2 model, but uses LoRA r=32/alpha=64.

set -euo pipefail

export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
export ORACLE_OUT_DIR="${ORACLE_OUT_DIR:-data/oracle_woz_v2}"
export OUTPUT_DIR="${OUTPUT_DIR:-models/v2/qwen2_5_3b_oracle_woz_v2_r32}"
export LORA_R="${LORA_R:-32}"
export LORA_ALPHA="${LORA_ALPHA:-64}"
export LR="${LR:-2e-4}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
export EVAL_STEPS="${EVAL_STEPS:-100}"
export DISABLE_EVAL="${DISABLE_EVAL:-0}"
export SEED="${SEED:-43}"

exec bash unity_config/job_oracle_all.sbatch
