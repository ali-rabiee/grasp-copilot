#!/bin/bash
#SBATCH --job-name=paper_bench_valid
#SBATCH --partition=gpu,uri-gpu
#SBATCH --gpus=1
#SBATCH --constraint=l40s|a100|h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/slurm/%x_%j.pre.log
#SBATCH --error=logs/slurm/%x_%j.pre.log

# Valid-set paper benchmark for the current data layout.
#
# Evaluates the trained LoRAs and heuristic baselines on:
#   data/oracle/reach_to_grasp/llm_contract_valid.jsonl
#   data/oracle/stacking/llm_contract_valid.jsonl
#   data/oracle/pouring/llm_contract_valid.jsonl
#   data/woz/all/llm_contract_valid.jsonl
#
# Default submit, after the model directories are ready:
#   sbatch unity_config/job_paper_benchmark.sbatch
#
# Useful overrides:
#   MODELS=oracle_lora,oracle_ycb,oracle_stacking,oracle_pouring sbatch unity_config/job_paper_benchmark.sbatch
#   OUT_DIR=evaluation/results/paper_benchmark sbatch unity_config/job_paper_benchmark.sbatch
#   RERUN=1 sbatch unity_config/job_paper_benchmark.sbatch

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-copilot}"
OUT_DIR="${OUT_DIR:-evaluation/results/paper_benchmark_valid}"
MODELS="${MODELS:-}"
EVAL_SETS="${EVAL_SETS:-oracle_valid_ycb,oracle_valid_stacking,oracle_valid_pouring,woz_valid}"
INCLUDE_ZERO_SHOT="${INCLUDE_ZERO_SHOT:-0}"
SKIP_HEURISTICS="${SKIP_HEURISTICS:-0}"
USE_4BIT="${USE_4BIT:-0}"
RERUN="${RERUN:-0}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"
SEED="${SEED:-0}"
REFRESH_EXISTING_SUMMARY="${REFRESH_EXISTING_SUMMARY:-1}"

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"

if [ ! -f "evaluation/benchmarks/run_paper_benchmark.py" ]; then
    echo "Run this job from the grasp-copilot repo root."
    echo "Current directory: $REPO"
    exit 2
fi

for f in \
    data/oracle/reach_to_grasp/llm_contract_valid.jsonl \
    data/oracle/stacking/llm_contract_valid.jsonl \
    data/oracle/pouring/llm_contract_valid.jsonl \
    data/woz/all/llm_contract_valid.jsonl
do
    if [ ! -f "$f" ]; then
        echo "Required eval file not found: $f"
        exit 2
    fi
done

_model_path_for_key() {
    case "$1" in
        oracle_lora) echo "models/qwen2_5_3b_oracle_lora" ;;
        woz_lora) echo "models/qwen2_5_3b_woz_lora" ;;
        oracle_woz_lora) echo "models/qwen2_5_3b_oracle_woz_lora" ;;
        oracle_woz_replay) echo "models/qwen2_5_3b_oracle_woz_lora_replay" ;;
        oracle_woz_replay_v2) echo "models/qwen2_5_3b_oracle_woz_lora_replay_v2" ;;
        oracle_woz_v2) echo "models/v2/qwen2_5_3b_oracle_woz_v2" ;;
        oracle_woz_v2_ycb) echo "models/v2/qwen2_5_3b_oracle_woz_v2_ycb" ;;
        oracle_woz_v2_stacking) echo "models/v2/qwen2_5_3b_oracle_woz_v2_stacking" ;;
        oracle_woz_v2_pouring) echo "models/v2/qwen2_5_3b_oracle_woz_v2_pouring" ;;
        oracle_woz_v2_r32) echo "models/v2/qwen2_5_3b_oracle_woz_v2_r32" ;;
        oracle_woz_r32) echo "models/qwen2_5_3b_oracle_woz_lora_r32" ;;
        oracle_ycb) echo "models/qwen2_5_3b_oracle_lora_ycb" ;;
        oracle_stacking) echo "models/qwen2_5_3b_oracle_lora_stacking" ;;
        oracle_pouring) echo "models/qwen2_5_3b_oracle_lora_pouring" ;;
        *) return 1 ;;
    esac
}

_required_model_keys="$MODELS"
if [ -z "$_required_model_keys" ]; then
    _required_model_keys="oracle_lora,woz_lora,oracle_woz_lora,oracle_woz_replay,oracle_woz_r32,oracle_ycb,oracle_stacking,oracle_pouring"
fi

IFS=',' read -r -a _keys <<< "$_required_model_keys"
for key in "${_keys[@]}"; do
    key="${key//[[:space:]]/}"
    [ -n "$key" ] || continue
    if path="$(_model_path_for_key "$key")"; then
        if [ ! -d "$path" ]; then
            echo "Model dir not found for $key: $path"
            echo "Either finish/copy that model first or set MODELS to a ready subset."
            exit 2
        fi
        if [ ! -f "$path/config.json" ]; then
            echo "Model dir missing config.json for $key: $path"
            exit 2
        fi
        if ! compgen -G "$path/model*.safetensors" >/dev/null && [ ! -f "$path/pytorch_model.bin" ]; then
            echo "Model dir missing weights for $key: $path"
            exit 2
        fi
    fi
done

JOB_NAME="${SLURM_JOB_NAME:-paper_bench_valid}"
JOB_ID="${SLURM_JOB_ID:-manual}"
mkdir -p "logs/slurm" "$OUT_DIR"
exec >>"logs/slurm/${JOB_NAME}_${JOB_ID}.log" 2>&1

echo "[job] started at $(date --iso-8601=seconds)"
echo "[job] node=${SLURM_NODELIST:-unknown} job_id=$JOB_ID"
echo "[job] repo=$REPO"
echo "[job] out_dir=$OUT_DIR"
echo "[job] models=${MODELS:-(default all trained LoRAs)}"
echo "[job] eval_sets=$EVAL_SETS"
echo "[job] include_zero_shot=$INCLUDE_ZERO_SHOT skip_heuristics=$SKIP_HEURISTICS"
echo "[job] max_examples=$MAX_EXAMPLES max_new_tokens=$MAX_NEW_TOKENS rerun=$RERUN"
echo "[job] refresh_existing_summary=$REFRESH_EXISTING_SUMMARY"

if [ -f /modules/apps/conda/main/etc/profile.d/conda.sh ]; then
    source /modules/apps/conda/main/etc/profile.d/conda.sh
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

_ARGS=(--out_dir "$OUT_DIR" --eval_sets "$EVAL_SETS" --seed "$SEED" --max_new_tokens "$MAX_NEW_TOKENS" --progress_every "$PROGRESS_EVERY")

if [ -n "$MODELS" ]; then
    _ARGS+=(--models "$MODELS")
fi
if [ "$MAX_EXAMPLES" -gt 0 ]; then
    _ARGS+=(--max_examples "$MAX_EXAMPLES")
fi
if [ "$INCLUDE_ZERO_SHOT" = "1" ] || [ "$INCLUDE_ZERO_SHOT" = "true" ]; then
    _ARGS+=(--include_zero_shot)
fi
if [ "$SKIP_HEURISTICS" = "1" ] || [ "$SKIP_HEURISTICS" = "true" ]; then
    _ARGS+=(--skip_heuristics)
fi
if [ "$USE_4BIT" = "1" ] || [ "$USE_4BIT" = "true" ]; then
    _ARGS+=(--use_4bit)
fi
if [ "$RERUN" = "1" ] || [ "$RERUN" = "true" ]; then
    _ARGS+=(--rerun)
fi

python -m evaluation.benchmarks.run_paper_benchmark "${_ARGS[@]}"

if [ "$REFRESH_EXISTING_SUMMARY" = "1" ] || [ "$REFRESH_EXISTING_SUMMARY" = "true" ]; then
    python -m evaluation.benchmarks.refresh_summary_from_results --out_dir "$OUT_DIR"
fi

echo "[job] finished at $(date --iso-8601=seconds)"
