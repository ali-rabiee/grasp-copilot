#!/bin/bash
#SBATCH --job-name=paper_post_valid
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --open-mode=append

# Build paper tables and figures from a completed valid-set paper benchmark.
#
# Default submit:
#   sbatch unity_config/job_paper_postprocess.sbatch
#
# Uses the same default OUT_DIR as job_paper_benchmark.sbatch.

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-copilot}"
OUT_DIR="${OUT_DIR:-evaluation/results/paper_benchmark_valid}"
INCLUDE_ZERO_SHOT="${INCLUDE_ZERO_SHOT:-0}"
BUILD_FIGURES="${BUILD_FIGURES:-1}"
FIGURE_EVAL_SET="${FIGURE_EVAL_SET:-oracle_valid_ycb}"

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"

if [ ! -f "$OUT_DIR/manifest.json" ]; then
    echo "Benchmark manifest not found: $OUT_DIR/manifest.json"
    echo "Run unity_config/job_paper_benchmark.sbatch first."
    exit 2
fi

JOB_NAME="${SLURM_JOB_NAME:-paper_post_valid}"
JOB_ID="${SLURM_JOB_ID:-manual}"
mkdir -p "logs/slurm" "$OUT_DIR/logs"
exec >>"logs/slurm/${JOB_NAME}_${JOB_ID}.log" 2>&1

echo "[job] started at $(date --iso-8601=seconds)"
echo "[job] repo=$REPO"
echo "[job] out_dir=$OUT_DIR"
echo "[job] include_zero_shot=$INCLUDE_ZERO_SHOT build_figures=$BUILD_FIGURES figure_eval_set=$FIGURE_EVAL_SET"

if [ -f /modules/apps/conda/main/etc/profile.d/conda.sh ]; then
    source /modules/apps/conda/main/etc/profile.d/conda.sh
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

_TABLE_ARGS=(--bench_dir "$OUT_DIR")
if [ "$INCLUDE_ZERO_SHOT" = "1" ] || [ "$INCLUDE_ZERO_SHOT" = "true" ]; then
    _TABLE_ARGS+=(--include_zs)
fi

python -m evaluation.tables.build_paper_tables "${_TABLE_ARGS[@]}" 2>&1 | tee "$OUT_DIR/logs/tables_${JOB_ID}.log"

if [ "$BUILD_FIGURES" = "1" ] || [ "$BUILD_FIGURES" = "true" ]; then
    python -m evaluation.plots.paper_figures \
        --bench_dir "$OUT_DIR" \
        --eval_set "$FIGURE_EVAL_SET" \
        2>&1 | tee "$OUT_DIR/logs/figures_${JOB_ID}.log"
fi

echo "[job] finished at $(date --iso-8601=seconds)"
