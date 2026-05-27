#!/bin/bash
#SBATCH --job-name=grasp_combo_pour
#SBATCH --partition=gpu,uri-gpu
#SBATCH --gpus=1
#SBATCH --constraint=l40s|a100|h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --open-mode=append

# Independent pouring Oracle+WoZ model trained from the base Qwen model.

set -euo pipefail

export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
export ORACLE_OUT_DIR="${ORACLE_OUT_DIR:-data/oracle_woz_v2_pouring}"
export OUTPUT_DIR="${OUTPUT_DIR:-models/v2/qwen2_5_3b_oracle_woz_v2_pouring}"
export LR="${LR:-2e-4}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
export EVAL_STEPS="${EVAL_STEPS:-100}"
export DISABLE_EVAL="${DISABLE_EVAL:-0}"
export SEED="${SEED:-41}"

exec bash unity_config/job_oracle_all.sbatch
