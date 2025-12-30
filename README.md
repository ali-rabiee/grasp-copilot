# grasp-copilot
a copilot llm for suggesting grasp assistives.

## Installation

This project uses a Conda environment named **`llm`**.

### One-command install (recommended)

From the repo root:

```bash
bash grasp-copilot/install_conda.sh
conda activate llm
```

- **CPU-only PyTorch (default)**: `bash grasp-copilot/install_conda.sh --cpu`
- **CUDA PyTorch (NVIDIA)**: `bash grasp-copilot/install_conda.sh --cuda 12.1`
- **Different env name**: `bash grasp-copilot/install_conda.sh -n grasp-copilot-llm`
- **Recreate env**: `bash grasp-copilot/install_conda.sh --force`

Quick sanity check:

```bash
conda activate llm
python -m pytest -q grasp-copilot/llm/tests
```

### Manual install (if you prefer)

```bash
conda env create -f grasp-copilot/environment.yml
conda activate llm

# Install torch in the way that matches your machine (CPU vs CUDA),
# then install the rest:
python -m pip install -r grasp-copilot/requirements.txt
```

## Dataset generation + inspection

All commands must activate the `llm` environment first:

```bash
conda activate llm
```

### Generate a dataset (raw generator JSONL)

```bash
conda activate llm
python -m data_generator.generate_dataset --episodes 1000 --seed 0 --out /tmp/grasp_gen.jsonl
# also writes: /tmp/grasp_gen.jsonl.stats.json
```

### One-shot: collect (scripted) + prepare LLM training data

This runs the scripted generator and immediately produces both the dataset-contract JSONL and
Qwen chat JSONL for training.

```bash
conda activate llm
python -m data_generator.collect_and_prepare --episodes 1000 --seed 0 --out_dir /tmp/grasp_run
# writes:
#   /tmp/grasp_run/grasp_gen.jsonl (+ .stats.json)
#   /tmp/grasp_run/llm_contract.jsonl
#   /tmp/grasp_run/llm_chat.jsonl
```

### Inspect the raw generator dataset (recommended for `/tmp/grasp_gen.jsonl`)

The generator records contain keys like `episode_id`, `objects`, `gripper_hist`, `memory`, `user_state`, `target_tool_call`.

```bash
conda activate llm
python -m data_generator.inspect_data --path /tmp/grasp_gen.jsonl --summary
python -m data_generator.inspect_data --path /tmp/grasp_gen.jsonl --episode 0 --max-t 20 --show-objects --show-gripper --show-memory
```

### Convert generator JSONL → LLM contract/chat JSONL, then inspect

```bash
conda activate llm
python scripts/prepare_llm_data.py --generator_jsonl /tmp/grasp_gen.jsonl --out_contract /tmp/grasp_contract.jsonl --out_chat /tmp/grasp_chat.jsonl

python scripts/inspect_data.py --file /tmp/grasp_contract.jsonl --mode contract --n 3
python scripts/inspect_data.py --file /tmp/grasp_chat.jsonl --mode chat --n 1
```

Note: `python scripts/inspect_data.py --mode generator` expects different fields (`obs`, `dialog`) than the raw generator output; for raw generator inspection use `python -m data_generator.inspect_data`.

### GUI playground (visual validation + interactive assistance)

This opens a small white-canvas GUI where you can:
- Move the gripper with the keyboard (arrows + yaw/z keys)
- Click **Ask assistance** to query either the oracle backend or your HF model
- Click a choice button to send the user response back into memory

```bash
conda activate llm
python scripts/gui_assist_demo.py --backend oracle
```

HF backend (example):

```bash
conda activate llm
python scripts/gui_assist_demo.py --backend hf --model_path outputs/qwen_merged_005
```

## LLM fine-tuning + inference

All commands must activate the `llm` environment first:

```bash
conda activate llm
```

### Prepare LLM data (for training a merged model)

```bash
conda activate llm
python scripts/prepare_llm_data.py --generator_jsonl data.jsonl --out_contract llm_contract.jsonl --out_chat llm_chat.jsonl
```

### Train SFT + LoRA / QLoRA

All commands must activate the `llm` environment first:

```bash
conda activate llm
python scripts/train_sft_lora.py --help
```

#### Quick recipes

- **Smoke run (a few steps, no eval, no checkpoints)**

```bash
conda activate llm
python scripts/train_sft_lora.py \
  --train_path data/runs/005/llm_contract.jsonl \
  --output_dir outputs/qwen_merged_smoke_005 \
  --max_steps 20 \
  --logging_steps 1 \
  --save_strategy no \
  --disable_eval \
  --no-packing \
  --max_grad_norm 1.0
```

- **Typical training run (with eval, still memory-safe defaults)**

```bash
conda activate llm
python scripts/train_sft_lora.py \
  --train_path data/runs/005/llm_contract.jsonl \
  --valid_path data/runs/005/llm_contract.jsonl \
  --output_dir outputs/qwen_merged_005 \
  --no-packing \
  --optim paged_adamw_32bit \
  --max_grad_norm 1.0
```

- **If you hit CUDA OOM**
  - Reduce context: `--max_seq_length 512` (or `256`)
  - Disable eval: `--disable_eval` (evaluation is often the OOM trigger)
  - Keep QLoRA: ensure you did not pass `--no-use_4bit`

#### `train_sft_lora.py` arguments (what they do)

- **Data / I/O**
  - **`--train_path`**: Path to the *contract* JSONL used for training (see “Prepare LLM data”).
  - **`--valid_path`**: Optional contract JSONL used for evaluation (ignored if `--disable_eval`).
  - **`--output_dir`**: Where the final **merged standalone model** is written at the end of training.

- **Model / memory**
  - **`--model_name`**: Base HF model id (default: `Qwen/Qwen2.5-7B-Instruct`).
  - **`--use_4bit` / `--no-use_4bit`**: Enable/disable 4-bit loading (QLoRA). 4-bit is the most reliable way to fit 7B models on ~16GB GPUs.
  - **`--max_seq_length`**: Max sequence length for SFT (bigger = more VRAM; if OOM, reduce).

- **Training loop**
  - **`--per_device_train_batch_size`**: Micro-batch size on each GPU.
  - **`--gradient_accumulation_steps`**: Accumulate gradients to simulate a larger batch without increasing VRAM.
  - **`--lr`**: Learning rate.
  - **`--num_train_epochs`**: Number of passes over the dataset (ignored if `--max_steps > 0`).
  - **`--max_steps`**: If `> 0`, train for exactly this many optimizer steps (best for quick tests).
  - **`--seed`**: Random seed.

- **LoRA config**
  - **`--lora_r`**, **`--lora_alpha`**, **`--lora_dropout`**: Standard LoRA hyperparameters.

- **Evaluation (memory sensitive)**
  - **`--disable_eval`**: Turn evaluation off entirely (ignores `--valid_path`).
  - **`--per_device_eval_batch_size`**: Eval micro-batch size (keep at `1` if close to VRAM limit).
  - **`--eval_steps`**: Evaluate every N steps (only used when eval is enabled).
  - **`--eval_accumulation_steps`**: Reduces peak memory when evaluating large sets.
  - **`--prediction_loss_only` / `--no-prediction_loss_only`**: If enabled, avoids storing full logits during eval (much lower VRAM).

- **Checkpointing / logging**
  - **`--save_strategy {no,steps,epoch}`**: Whether to save intermediate checkpoints.
    - Recommended: `no` (the final merged model is saved at the end anyway).
  - **`--save_steps`**: Steps between checkpoints (when `save_strategy=steps`).
  - **`--save_only_model`**: Save only model weights (avoids huge optimizer checkpoints).
  - **`--save_total_limit`**: Keep at most this many checkpoints.
  - **`--logging_steps`**: Log every N steps.
  - **`--warmup_ratio`**: LR warmup ratio.
  - **`--report_to`**: Logging backend (default: `none`).

### Merge LoRA (optional)

Note: newer training runs already write a merged model directly (no manual merge step needed).

### Inference demo (JSON-only)

```bash
conda activate llm
python scripts/inference_demo.py --model_path outputs/qwen_merged_005 --prompt 'Return {"tool_name":"INTERACT","arguments":{"type":"notify","text":"ok"}}'
```
