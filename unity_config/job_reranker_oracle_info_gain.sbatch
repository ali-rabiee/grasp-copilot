#!/bin/bash
#SBATCH --job-name=rerank_oracle_ig
#SBATCH --partition=gpu,uri-gpu
#SBATCH --gpus=1
#SBATCH --constraint=l40s|a100|h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --open-mode=append

# Oracle-only baseline with the information-gain reranker.

export MODEL_KEY="${MODEL_KEY:-oracle_lora}"
export MODEL_PATH="${MODEL_PATH:-models/qwen2_5_3b_oracle_lora}"
export RERANK_MODE="${RERANK_MODE:-info_gain}"

exec bash unity_config/job_reranker_ablation.sbatch
