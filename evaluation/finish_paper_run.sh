#!/usr/bin/env bash
# Post-benchmark finishing script.
#
# Run this AFTER the main paper benchmark has finished
# (`evaluation/run_paper_benchmark.py` is no longer running and the manifest
# exists). It will:
#
#   1. Optionally add a zero-shot Qwen2.5-3B-Instruct reference row.
#   2. Run the GPU-bound LLM robustness sweep (Oracle/H1 reference curves are
#      already cached if you ran --baselines_only earlier).
#   3. Regenerate paper tables + figures from the full cache.
#
# Usage:
#   bash evaluation/finish_paper_run.sh           # full finish (~3 h GPU)
#   bash evaluation/finish_paper_run.sh --skip-zs # skip zero-shot
#   bash evaluation/finish_paper_run.sh --skip-robust  # skip robustness
#   bash evaluation/finish_paper_run.sh --tables-only  # just rebuild outputs

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="evaluation/eval_outputs/paper_benchmark"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

SKIP_ZS=0
SKIP_ROBUST=0
TABLES_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --skip-zs)       SKIP_ZS=1 ;;
        --skip-robust)   SKIP_ROBUST=1 ;;
        --tables-only)   TABLES_ONLY=1 ; SKIP_ZS=1 ; SKIP_ROBUST=1 ;;
        -h|--help)
            grep '^# ' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown arg: $arg"; exit 2 ;;
    esac
done

# Sanity: confirm main benchmark is done.
if pgrep -fa "run_paper_benchmark" >/dev/null; then
    echo "[finish] Main benchmark is still running. Wait for it to finish, then re-run this."
    pgrep -fa "run_paper_benchmark"
    exit 1
fi

if [ ! -f "$OUT_DIR/manifest.json" ]; then
    echo "[finish] Manifest not found at $OUT_DIR/manifest.json — main benchmark hasn't completed yet."
    exit 1
fi

n_cells=$(ls "$OUT_DIR/results" 2>/dev/null | wc -l)
echo "[finish] Found $n_cells cached cells. Starting finishing pass."

if [ $SKIP_ZS -eq 0 ]; then
    echo ""
    echo "############################################################"
    echo "# Step 1/3: Adding zero-shot Qwen2.5-3B-Instruct reference"
    echo "############################################################"
    echo "[finish] Downloading Qwen/Qwen2.5-3B-Instruct on first hit (~7 GB)..."
    python -m evaluation.run_paper_benchmark \
        --include_zero_shot --skip_trained --skip_heuristics \
        2>&1 | tee "$LOG_DIR/zs.log"
fi

if [ $SKIP_ROBUST -eq 0 ]; then
    echo ""
    echo "############################################################"
    echo "# Step 2/3: LLM robustness sweep (7 LoRAs × 3 envs × 5 noise)"
    echo "############################################################"
    echo "[finish] This is ~2–3 h of GPU work."
    python -m evaluation.run_robustness_sweep \
        --max_examples 300 \
        2>&1 | tee "$LOG_DIR/robustness.log"
fi

echo ""
echo "############################################################"
echo "# Step 3/3: Building final tables + figures"
echo "############################################################"
python -m evaluation.tables.build_paper_tables \
    ${ZS_FLAG:+--include_zs} \
    2>&1 | tee "$LOG_DIR/tables.log"

# Generate figures keyed to two representative eval sets so the paper has both
# the 'easy' (YCB) and 'hard' (Pouring) confusion / context / radar variants.
for es in oracle_valid_ycb oracle_valid_pouring; do
    echo ""
    echo "[finish] figures focused on $es ..."
    python -m evaluation.plots.paper_figures \
        --eval_set "$es" \
        --out_dir "$OUT_DIR/figures_$es" \
        2>&1 | tee -a "$LOG_DIR/figures.log"
done

# Run the standard figure set (overwriting the YCB-keyed one to be the
# canonical 'figures/' directory).
python -m evaluation.plots.paper_figures \
    --eval_set oracle_valid_ycb 2>&1 | tee -a "$LOG_DIR/figures.log"

# Plot robustness curves (no-op if no sweep was run).
python -m evaluation.run_robustness_sweep --plot_only \
    2>&1 | tee -a "$LOG_DIR/figures.log"

echo ""
echo "############################################################"
echo "# DONE"
echo "############################################################"
echo "Tables:   $OUT_DIR/tables/"
echo "Figures:  $OUT_DIR/figures/"
echo "          $OUT_DIR/figures_oracle_valid_ycb/"
echo "          $OUT_DIR/figures_oracle_valid_pouring/"
echo "Robustness: $OUT_DIR/robustness/robustness_curves.pdf"
echo "Logs:     $LOG_DIR/"
