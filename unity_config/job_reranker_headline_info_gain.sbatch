#!/bin/bash
#SBATCH --job-name=rerank_comb_ig
#SBATCH --partition=gpu,uri-gpu
#SBATCH --gpus=1
#SBATCH --constraint=l40s|a100|h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --open-mode=append

# Headline v2 model with the information-gain reranker enabled.

export MODEL_KEY="${MODEL_KEY:-oracle_woz_v2}"
export MODEL_PATH="${MODEL_PATH:-models/v2/qwen2_5_3b_oracle_woz_v2}"
export RERANK_MODE="${RERANK_MODE:-info_gain}"

exec bash unity_config/job_reranker_ablation.sbatch
