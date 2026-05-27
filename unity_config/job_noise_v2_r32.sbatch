#!/bin/bash
#SBATCH --job-name=noise_comb_r32
#SBATCH --partition=gpu,uri-gpu
#SBATCH --gpus=1
#SBATCH --constraint=l40s|a100|h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --open-mode=append

# Rank-32 v2-model ablation for the user-input-noise sweep.

export MODEL_KEY="${MODEL_KEY:-oracle_woz_v2_r32}"
export MODEL_PATH="${MODEL_PATH:-models/v2/qwen2_5_3b_oracle_woz_v2_r32}"

exec bash unity_config/job_noise_sweep.sbatch
