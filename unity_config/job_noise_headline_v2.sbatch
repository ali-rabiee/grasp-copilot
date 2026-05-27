#!/bin/bash
#SBATCH --job-name=noise_v2
#SBATCH --partition=gpu,uri-gpu
#SBATCH --gpus=1
#SBATCH --constraint=l40s|a100|h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --open-mode=append

# Headline user-input-noise sweep for the independent Oracle+WoZ v2 model.
# Uses the committed 141-scenario real-user rollout corpus; no training JSONL is
# needed for this evaluation.

export MODEL_KEY="${MODEL_KEY:-oracle_woz_v2}"
export MODEL_PATH="${MODEL_PATH:-models/v2/qwen2_5_3b_oracle_woz_v2}"

exec bash unity_config/job_noise_sweep.sbatch
