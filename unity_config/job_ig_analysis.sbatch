#!/bin/bash
#SBATCH --job-name=ig_analysis
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --open-mode=append

# Post-hoc IG analysis on logged reranker dialogs.
# (see plans/package4_ig_reranker.md §9.2).
#
# Reads every dialogs.jsonl under $ABLATION_ROOT and produces:
#   $OUT_DIR/per_question.csv     (one row per dialog × selector)
#   $OUT_DIR/summary.json         (aggregate means, by-kind splits)
#   $OUT_DIR/ig_distribution.{pdf,png}
#   $OUT_DIR/ig_by_kind.{pdf,png}
# Then re-builds the headline ablation table from rollouts + IG data.
#
# CPU-only, finishes in minutes. Run after the GPU sweeps complete.
#
# Tunable env vars:
#   ABLATION_ROOT   default: evaluation/results/reranker/ablation
#   OUT_DIR         default: evaluation/results/reranker/ig_analysis
#   TABLES_DIR      default: evaluation/results/reranker/tables
#   CONDA_ENV       default: copilot

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-copilot}"
ABLATION_ROOT="${ABLATION_ROOT:-evaluation/results/reranker/ablation}"
OUT_DIR="${OUT_DIR:-evaluation/results/reranker/ig_analysis}"
TABLES_DIR="${TABLES_DIR:-evaluation/results/reranker/tables}"

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"

[ -d "$ABLATION_ROOT" ] || { echo "Ablation root not found: $ABLATION_ROOT"; exit 2; }

JOB_ID="${SLURM_JOB_ID:-manual}"
mkdir -p logs/slurm "$OUT_DIR" "$TABLES_DIR"
exec >>"logs/slurm/ig_analysis_${JOB_ID}.log" 2>&1

echo "[job] started at $(date --iso-8601=seconds)"
echo "[job] ablation_root=$ABLATION_ROOT"
echo "[job] out_dir=$OUT_DIR"
echo "[job] tables_dir=$TABLES_DIR"

if [ -f /modules/apps/conda/main/etc/profile.d/conda.sh ]; then
    source /modules/apps/conda/main/etc/profile.d/conda.sh
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

python -m evaluation.reranker.analyze_ig \
    --ablation_root "$ABLATION_ROOT" \
    --out_dir       "$OUT_DIR"

python -m evaluation.reranker.plot_ig \
    --out_dir "$OUT_DIR"

python -m evaluation.reranker.tables_reranker \
    --ablation_root  "$ABLATION_ROOT" \
    --ig_analysis_dir "$OUT_DIR" \
    --out_dir        "$TABLES_DIR"

echo "[job] finished at $(date --iso-8601=seconds)"
