#!/bin/bash
#SBATCH --job-name=grasp_p2_woz_v2
#SBATCH --partition=gpu,uri-gpu
#SBATCH --gpus=1
#SBATCH --constraint=l40s|a100|h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --open-mode=append

# Stronger Oracle->WoZ replay experiment:
# - all oracle train rows are retained
# - WoZ train is repeated to give the new supervision enough weight
# - LR/epochs are reduced to avoid overwriting oracle formatting/templates

set -euo pipefail

export PHASE2_OUT_DIR="${PHASE2_OUT_DIR:-data/oracle_woz_replay_v2}"
export OUTPUT_DIR="${OUTPUT_DIR:-models/qwen2_5_3b_oracle_woz_lora_replay_v2}"
export LR="${LR:-1e-5}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export EVAL_STEPS="${EVAL_STEPS:-50}"
export DISABLE_EVAL="${DISABLE_EVAL:-0}"
export SEED="${SEED:-17}"

exec bash unity_config/job.sbatch
