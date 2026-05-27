#!/usr/bin/env bash
# Run a strong-LLM zero-shot benchmark on the four CoRL eval sets.
#
# Use when network can sustain a ~15 GB Qwen2.5-7B-Instruct download or
# point to a locally cached model directory. With --use_4bit the model
# fits on a 12 GB GPU; on larger GPUs drop the flag for BF16 inference.
#
# Outputs:
#   evaluation/results/paper_benchmark/per_model_results/qwen7b_zs__*.json
#   evaluation/results/paper_benchmark/results/qwen7b_zs__*.json
#
# After running, re-run evaluation/benchmarks/refresh_summary_from_results.py
# to regenerate summary_all.csv with the new row.

set -e
cd "$(dirname "$0")/.."

MODEL_NAME="${MODEL_NAME:-qwen7b_zs}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
OUT_DIR="${OUT_DIR:-evaluation/results/paper_benchmark/per_model_results}"
FLAGS="${EXTRA_FLAGS:---use_4bit}"

EVAL_SETS=(
  "data/oracle_valid_ycb/llm_contract_200.jsonl"
  "data/oracle_valid_stacking/llm_contract_200.jsonl"
  "data/oracle_valid_pouring/llm_contract_200.jsonl"
  "data/woz_phase2/llm_contract_valid.jsonl"
)

for path in "${EVAL_SETS[@]}"; do
  echo "=== Running $MODEL_NAME on $path ==="
  python3 -m evaluation.benchmarks.offline_exec_benchmark \
    --contract_jsonl "$path" \
    --models "${MODEL_NAME}=${MODEL_PATH}" \
    --out_dir "$OUT_DIR" \
    --progress_every 25 \
    $FLAGS
done

echo "Done. Re-aggregate with:"
echo "  python3 evaluation/benchmarks/refresh_summary_from_results.py"
