#!/bin/bash
#SBATCH --job-name=noise_oracle
#SBATCH --partition=gpu,uri-gpu
#SBATCH --gpus=1
#SBATCH --constraint=l40s|a100|h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --open-mode=append

# Oracle-only LLM baseline for the user-input-noise sweep.

export MODEL_KEY="${MODEL_KEY:-oracle_lora}"
export MODEL_PATH="${MODEL_PATH:-models/qwen2_5_3b_oracle_lora}"

exec bash unity_config/job_noise_sweep.sbatch
