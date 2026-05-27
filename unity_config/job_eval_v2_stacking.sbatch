#!/bin/bash
#SBATCH --job-name=eval_combo_stack
#SBATCH --partition=gpu,uri-gpu
#SBATCH --gpus=1
#SBATCH --constraint=l40s|a100|h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --open-mode=append

set -euo pipefail

export MODELS="${MODELS:-oracle_woz_v2_stacking}"
export SKIP_HEURISTICS="${SKIP_HEURISTICS:-1}"
export OUT_DIR="${OUT_DIR:-evaluation/results/paper_benchmark_v2}"
export RERUN="${RERUN:-1}"
export REFRESH_EXISTING_SUMMARY="${REFRESH_EXISTING_SUMMARY:-1}"

exec bash unity_config/job_paper_benchmark.sbatch
