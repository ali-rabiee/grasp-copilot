#!/bin/bash
#SBATCH --job-name=noise_sweep
#SBATCH --partition=gpu,uri-gpu
#SBATCH --gpus=1
#SBATCH --constraint=l40s|a100|h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --open-mode=append

# User-input-noise robustness sweep
# (see evaluation/results/robustness/user_input_noise/README.md).
#
# Each submission runs the (160 scenarios × 6 conditions × N seeds) matrix
# for ONE model or backend. Submit one job per model; jobs can run in
# parallel on different Unity nodes.
#
# Examples (from grasp-copilot repo root):
#
#   # Headline LLM:
#   MODEL_KEY=oracle_woz_lora MODEL_PATH=models/qwen2_5_3b_oracle_woz_lora \
#       sbatch unity_config/job_noise_sweep.sbatch
#
#   # Warm-start ablation:
#   MODEL_KEY=oracle_lora MODEL_PATH=models/qwen2_5_3b_oracle_lora \
#       sbatch unity_config/job_noise_sweep.sbatch
#
#   # Pure-WoZ control:
#   MODEL_KEY=woz_lora MODEL_PATH=models/qwen2_5_3b_woz_lora \
#       sbatch unity_config/job_noise_sweep.sbatch
#
#   # Heuristic baseline (CPU only, but uses the same GPU node — fine, finishes fast):
#   BACKEND=heuristic HEURISTIC=h1_ask_if_amb \
#       sbatch unity_config/job_noise_sweep.sbatch
#
#   # Manual baseline:
#   BACKEND=manual MODES=manual \
#       sbatch unity_config/job_noise_sweep.sbatch
#
# Tunable env vars (with defaults):
#   BACKEND        hf_ft | manual | defer | oracle | heuristic   (default: hf_ft)
#   MODEL_KEY      short name for hf_ft sweep                    (required when BACKEND=hf_ft)
#   MODEL_PATH     path to fine-tuned model                      (required when BACKEND=hf_ft)
#   HEURISTIC      h1_ask_if_amb | h2_always_ask | sa1_pred_assist | sa2_bayes_intent
#                                                                (required when BACKEND=heuristic)
#   MODES          space-separated subset of "manual prime"      (default: "manual prime")
#   N_SEEDS        rollouts per (scenario, mode, condition)      (default: 5)
#   MAX_TICKS      hard cap on rollout length                    (default: 2000)
#   CONDITIONS     space-separated noise conditions              (default: all 6)
#   WORKERS        parallel processes for CPU backends           (default: 1; ignored for hf_ft)
#   CONDA_ENV      conda env name                                (default: copilot)
#
# Time budget: ~20 h is safe for an A100/L40S running the full hf_ft matrix
# (160 × 6 × 2 × 5 ≈ 9,600 rollouts, mostly LLM-bound). Drop to 4 h for
# manual/heuristic/oracle.

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-copilot}"
BACKEND="${BACKEND:-hf_ft}"
MODEL_KEY="${MODEL_KEY:-}"
MODEL_PATH="${MODEL_PATH:-}"
HEURISTIC="${HEURISTIC:-}"
MODES="${MODES:-manual prime}"
N_SEEDS="${N_SEEDS:-5}"
MAX_TICKS="${MAX_TICKS:-2000}"
CONDITIONS="${CONDITIONS:-clean dir_low dir_high sel_low sel_high compound_mid}"
WORKERS="${WORKERS:-1}"

SCENARIOS="evaluation/results/robustness/user_input_noise/scenarios/scenarios.labeled.jsonl"

# Decide sweep name + backend-specific extra args.
if [ "$BACKEND" = "hf_ft" ]; then
    [ -n "$MODEL_KEY" ]  || { echo "MODEL_KEY required for BACKEND=hf_ft";  exit 2; }
    [ -n "$MODEL_PATH" ] || { echo "MODEL_PATH required for BACKEND=hf_ft"; exit 2; }
    SWEEP_NAME="$MODEL_KEY"
    EXTRA_ARGS=(--backend hf_ft --model_paths "$MODEL_KEY=$MODEL_PATH" --workers 1)
elif [ "$BACKEND" = "heuristic" ]; then
    [ -n "$HEURISTIC" ] || { echo "HEURISTIC required for BACKEND=heuristic"; exit 2; }
    SWEEP_NAME="$HEURISTIC"
    EXTRA_ARGS=(--backend heuristic --heuristic "$HEURISTIC" --workers "$WORKERS")
else
    SWEEP_NAME="$BACKEND"
    EXTRA_ARGS=(--backend "$BACKEND" --workers "$WORKERS")
fi

OUT_DIR="evaluation/results/robustness/user_input_noise/sweeps/$SWEEP_NAME"

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"

if [ ! -f "$SCENARIOS" ]; then
    echo "Scenarios file not found: $SCENARIOS"
    echo "Run extract_from_prime_logs first; see"
    echo "  evaluation/results/robustness/user_input_noise/README.md"
    exit 2
fi

# Sanity-check model directory for hf_ft.
if [ "$BACKEND" = "hf_ft" ]; then
    if [ ! -d "$MODEL_PATH" ]; then
        echo "Model dir not found: $MODEL_PATH"
        echo "Transfer the merged model tree from your dev machine to Unity first."
        exit 2
    fi
    if ! compgen -G "$MODEL_PATH/model*.safetensors" >/dev/null && [ ! -f "$MODEL_PATH/pytorch_model.bin" ]; then
        echo "Model dir missing weights (no model*.safetensors or pytorch_model.bin): $MODEL_PATH"
        exit 2
    fi
fi

JOB_NAME="${SLURM_JOB_NAME:-noise_sweep}"
JOB_ID="${SLURM_JOB_ID:-manual}"
mkdir -p "logs/slurm" "$OUT_DIR"
exec >>"logs/slurm/${JOB_NAME}_${SWEEP_NAME}_${JOB_ID}.log" 2>&1

echo "[job] started at $(date --iso-8601=seconds)"
echo "[job] node=${SLURM_NODELIST:-unknown}  job_id=$JOB_ID"
echo "[job] repo=$REPO"
echo "[job] backend=$BACKEND  sweep_name=$SWEEP_NAME"
echo "[job] model_key=${MODEL_KEY:-(n/a)}  model_path=${MODEL_PATH:-(n/a)}"
echo "[job] modes=$MODES  n_seeds=$N_SEEDS  conditions=$CONDITIONS"
echo "[job] max_ticks=$MAX_TICKS  workers=$WORKERS"
echo "[job] out_dir=$OUT_DIR"

if [ -f /modules/apps/conda/main/etc/profile.d/conda.sh ]; then
    source /modules/apps/conda/main/etc/profile.d/conda.sh
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

python -m evaluation.benchmarks.scenario_noise_sweep \
    --scenarios "$SCENARIOS" \
    --out_dir   "$OUT_DIR" \
    --modes $MODES \
    --conditions $CONDITIONS \
    --n_seeds   "$N_SEEDS" \
    --max_ticks "$MAX_TICKS" \
    --progress_every 100 \
    "${EXTRA_ARGS[@]}"

echo "[job] finished at $(date --iso-8601=seconds)"
