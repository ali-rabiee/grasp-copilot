# Data Generation and Model Training

This guide describes the end-to-end `grasp-copilot` workflow for generating oracle data, preparing LLM datasets, training models, evaluating them, and storing trained model artifacts.

Run commands from the project root:

```bash
cd /home/ali/github/PRIME/grasp-copilot
conda activate llm
```

## Overview

The local workflow is:

1. Install `grasp-copilot` in editable mode.
2. Generate scripted oracle episodes with `grasp-collect`.
3. Convert raw generator data into contract JSONL and chat JSONL.
4. Train with `grasp-train` using contract JSONL.
5. Evaluate the merged model with `grasp-eval`.
6. Run inference or upload/download models with the helper scripts.

Important directories:

- `data_generator/`: scripted environments and oracle policies.
- `llm/`: dataset validation, conversion, training, inference, and evaluation.
- `scripts/`: utility wrappers and dataset/model helpers.
- `data/`: generated datasets. This directory is ignored by Git.
- `models/`: trained model outputs. This directory is ignored by Git.

Installed CLIs:

- `grasp-collect`: generate oracle data and prepare LLM JSONL files.
- `grasp-train`: train LoRA, QLoRA, or full fine-tuned models.
- `grasp-eval`: evaluate a merged model on contract JSONL.
- `grasp-infer`: run one-off model inference.

## Installation

### One-command Conda setup

For CUDA training:

```bash
bash install_conda.sh --cuda 12.1
conda activate llm
```

For CPU-only setup:

```bash
bash install_conda.sh
conda activate llm
```

### Manual setup

Create the environment:

```bash
conda create -n llm python=3.11 -y
conda activate llm
```

Install PyTorch. Pick one:

```bash
# CPU-only
conda install -y -c pytorch pytorch cpuonly

# CUDA example
conda install -y -c pytorch -c nvidia pytorch pytorch-cuda=12.1
```

Install the package:

```bash
python -m pip install -U pip
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e ".[test]"   # pytest
python -m pip install -e ".[qlora]"  # bitsandbytes for 4-bit QLoRA
python -m pip install -e ".[plot]"   # matplotlib for plots
```

Sanity checks:

```bash
python -m pytest -q llm/tests data_generator/tests
grasp-collect --help
grasp-train --help
```

## Data Formats

The project uses three JSONL formats.

### Generator JSONL

This is the raw simulator/oracle output. Each line contains one oracle step with keys such as:

- `episode_id`
- `objects`
- `gripper_hist`
- `memory`
- `user_state`
- `target_tool_call`

Use this format for oracle debugging or for re-preparing LLM data later.

### Contract JSONL

This is the canonical training and evaluation format. Each line has:

- `id`: example id.
- `instruction`: task instruction.
- `input`: compact JSON string containing scene state, gripper history, memory, and user state.
- `output`: JSON string containing exactly one target tool call.

Use contract JSONL with `grasp-train` and `grasp-eval`.

### Chat JSONL

This is a Qwen-style chat conversion of the contract examples. The current trainer builds chat messages internally from contract JSONL, so chat JSONL is mostly useful for inspection, debugging, and external trainers.

## Generate Oracle Data

`grasp-collect` is the main entrypoint. It can generate raw oracle records and prepare LLM data in one command.

Supported environments:

- `reach_to_grasp_ycb`
- `cube_stacking`
- `pouring`

### Quick dataset

```bash
grasp-collect \
  --env reach_to_grasp_ycb \
  --episodes 1000 \
  --seed 0
```

With no `--out_dir`, output goes into an auto-numbered environment directory:

```text
data/ycb/01/
data/stacking/01/
data/pouring/01/
```

Default output files include the episode count:

```text
grasp_gen_1000.jsonl
grasp_gen_1000.jsonl.stats.json
llm_contract_1000.jsonl
llm_chat_1000.jsonl
```

### Generate all environments

```bash
grasp-collect --env reach_to_grasp_ycb --episodes 10000 --seed 0 --rebalance
grasp-collect --env cube_stacking      --episodes 10000 --seed 1 --rebalance
grasp-collect --env pouring            --episodes 10000 --seed 2 --rebalance
```

### Explicit output directory

```bash
grasp-collect \
  --env pouring \
  --episodes 5000 \
  --seed 7 \
  --out_dir data/pouring/pilot_5k \
  --rebalance
```

### Raw records only

```bash
grasp-collect \
  --env cube_stacking \
  --episodes 1000 \
  --out_dir data/stacking/raw_debug \
  --skip_prepare
```

### Re-prepare an existing generator file

```bash
grasp-collect \
  --generator_jsonl data/ycb/01/grasp_gen_1000.jsonl \
  --episodes 1000 \
  --out_dir data/ycb/01
```

Direct converter form:

```bash
python -m llm.prepare_llm_data \
  --generator_jsonl data/ycb/01/grasp_gen_1000.jsonl \
  --out_contract data/ycb/01/llm_contract_1000.jsonl \
  --out_chat data/ycb/01/llm_chat_1000.jsonl
```

## Rebalance Data

The oracle data is usually heavy on `INTERACT` examples. Rebalancing repeats executable action examples and can downsample interaction examples.

Recommended starter command:

```bash
grasp-collect \
  --env reach_to_grasp_ycb \
  --episodes 10000 \
  --seed 0 \
  --rebalance
```

This writes:

```text
llm_contract_10000_rebalanced.jsonl
llm_chat_10000_rebalanced.jsonl
```

Custom balancing:

```bash
grasp-collect \
  --env reach_to_grasp_ycb \
  --episodes 10000 \
  --motion_repeat 4 \
  --interact_keep_prob 0.7 \
  --rebalance_seed 0
```

Recommended split:

```text
train: data/ycb/01/llm_contract_10000_rebalanced.jsonl
valid: data/ycb/01/llm_contract_10000.jsonl
```

## Build Matched Small Datasets

For controlled WOZ/oracle experiments, use:

```bash
python scripts/build_consistent_small_datasets.py \
  --woz_source data/woz_phase2/llm_contract_all.jsonl \
  --woz_out data/woz_consistent_small \
  --oracle_out data/oracle_consistent_small \
  --report_out data/consistent_small_similarity_report.json \
  --seed 7 \
  --oracle_episodes_per_env 2500
```

This writes train/valid contract and chat files under:

```text
data/woz_consistent_small/
data/oracle_consistent_small/
```

Split those datasets by environment:

```bash
python scripts/split_consistent_small_by_env.py \
  --woz_source data/woz_consistent_small \
  --oracle_source data/oracle_consistent_small \
  --out_root data
```

## Train Models

`grasp-train` consumes contract JSONL. It validates the file, converts rows into chat messages internally, trains with TRL SFT, saves adapter artifacts, and by default writes a merged standalone model.

If `--output_dir` is omitted, the trainer derives the output path from the model and data path. The merged model is the directory to use for evaluation, inference, GUI demos, and Hugging Face upload.

### QLoRA

QLoRA is the default because `--use_4bit` defaults to true. Install bitsandbytes first:

```bash
python -m pip install -e ".[qlora]"
```

Train Qwen 2.5 3B with QLoRA:

```bash
grasp-train \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --train_path data/ycb/01/llm_contract_10000_rebalanced.jsonl \
  --valid_path data/ycb/01/llm_contract_10000.jsonl \
  --output_dir models/qwen2_5_3b_ycb_lora \
  --max_seq_length 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --lr 2e-5 \
  --num_train_epochs 1 \
  --logging_steps 20 \
  --eval_steps 200 \
  --save_strategy no
```

Quick smoke run:

```bash
grasp-train \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --train_path data/ycb/01/llm_contract_1000_rebalanced.jsonl \
  --valid_path data/ycb/01/llm_contract_1000.jsonl \
  --output_dir models/smoke_qwen_0_5b \
  --max_steps 10 \
  --disable_eval \
  --max_seq_length 512
```

### LoRA without 4-bit

```bash
grasp-train \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --train_path data/ycb/01/llm_contract_10000_rebalanced.jsonl \
  --valid_path data/ycb/01/llm_contract_10000.jsonl \
  --output_dir models/qwen2_5_3b_ycb_lora_fp \
  --no-use_4bit \
  --max_seq_length 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --lr 2e-5
```

### Full fine-tuning

Full fine-tuning trains all weights and requires much more VRAM. It cannot be used with 4-bit quantization.

```bash
grasp-train \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --train_path data/ycb/01/llm_contract_10000_rebalanced.jsonl \
  --valid_path data/ycb/01/llm_contract_10000.jsonl \
  --output_dir models/qwen2_5_3b_ycb_full_ft \
  --full_finetune \
  --no-use_4bit \
  --max_seq_length 2048 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --lr 2e-5 \
  --num_train_epochs 1 \
  --save_strategy steps \
  --save_steps 500 \
  --eval_steps 500
```

### Resume training

Create checkpoints:

```bash
grasp-train \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --train_path data/ycb/01/llm_contract_10000_rebalanced.jsonl \
  --valid_path data/ycb/01/llm_contract_10000.jsonl \
  --output_dir models/qwen2_5_3b_ycb_lora \
  --save_strategy steps \
  --save_steps 500
```

Resume:

```bash
grasp-train \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --train_path data/ycb/01/llm_contract_10000_rebalanced.jsonl \
  --valid_path data/ycb/01/llm_contract_10000.jsonl \
  --output_dir models/qwen2_5_3b_ycb_lora \
  --resume_from_checkpoint models/qwen2_5_3b_ycb_lora_adapter/checkpoint-500
```

### Useful training flags

- `--model_name`: base Hugging Face model id or local model path.
- `--train_path`: contract JSONL used for training.
- `--valid_path`: optional validation contract JSONL.
- `--output_dir`: final merged model directory.
- `--adapter_dir`: LoRA adapter/checkpoint directory. Defaults to `<output_dir>_adapter`.
- `--use_4bit` / `--no-use_4bit`: enable or disable QLoRA.
- `--full_finetune`: train all weights. Requires `--no-use_4bit`.
- `--max_seq_length`: reduce this if GPU memory is tight.
- `--per_device_train_batch_size`: per-device microbatch size.
- `--gradient_accumulation_steps`: increases effective batch size.
- `--lr`: learning rate.
- `--max_steps`: short debug runs; overrides epochs when greater than zero.
- `--save_strategy`: `no`, `steps`, or `epoch`.
- `--merge_at_end` / `--no-merge_at_end`: write or skip the final merged model.

## Evaluate Models

Evaluate a merged model on held-out contract JSONL:

```bash
grasp-eval \
  --contract_jsonl data/ycb/01/llm_contract_10000.jsonl \
  --model_path models/qwen2_5_3b_ycb_lora \
  --max_examples 500 \
  --seed 0 \
  --dump_mistakes_jsonl eval_outputs/qwen2_5_3b_ycb_mistakes.jsonl \
  --dump_all_jsonl eval_outputs/qwen2_5_3b_ycb_all.jsonl
```

The evaluator reports JSON validity, schema validity, tool accuracy, object/action accuracy, and context-specific breakdowns.

## Run Inference

Use a merged local model:

```bash
grasp-infer \
  --model_path models/qwen2_5_3b_ycb_lora \
  --prompt 'Return {"tool":"INTERACT","args":{"kind":"QUESTION","text":"ok?","choices":["yes","no"]}}'
```

Use a Hugging Face Hub model id:

```bash
grasp-infer \
  --model_path YOUR_HF_USERNAME_OR_ORG/grasp-copilot-qwen2_5_3b_ycb_lora \
  --prompt 'Return {"tool":"INTERACT","args":{"kind":"QUESTION","text":"ok?","choices":["yes","no"]}}'
```

## GUI Demo

Oracle backend:

```bash
python scripts/gui_assist_demo.py --backend oracle
```

HF/local model backend:

```bash
python scripts/gui_assist_demo.py \
  --backend hf \
  --model_path models/qwen2_5_3b_ycb_lora
```

## Hugging Face Model Storage

Log in once:

```bash
huggingface-cli login
```

List local model directories:

```bash
python scripts/hf_model_store.py list --source-root models
```

Upload all model directories under a source root:

```bash
python scripts/hf_model_store.py upload \
  --source-root models \
  --namespace YOUR_HF_USERNAME_OR_ORG
```

Upload one model:

```bash
python scripts/hf_model_store.py upload \
  --source-root models \
  --model qwen2_5_3b_ycb_lora \
  --repo-id YOUR_HF_USERNAME_OR_ORG/grasp-copilot-qwen2_5_3b_ycb_lora
```

Download a model:

```bash
python scripts/hf_model_store.py download \
  --repo-id YOUR_HF_USERNAME_OR_ORG/grasp-copilot-qwen2_5_3b_ycb_lora \
  --target-dir models \
  --local-name qwen2_5_3b_ycb_lora
```

Use your real Hugging Face username or organization. The placeholder namespace will fail with a permissions error.

## Troubleshooting

### CUDA out of memory

Try:

```bash
--max_seq_length 512
--per_device_train_batch_size 1
--gradient_accumulation_steps 32
--disable_eval
--save_strategy no
```

### bitsandbytes missing

Install QLoRA extras:

```bash
python -m pip install -e ".[qlora]"
```

Or train without 4-bit:

```bash
grasp-train ... --no-use_4bit
```

### Full fine-tune with 4-bit

This is invalid:

```bash
grasp-train ... --full_finetune --use_4bit
```

Use:

```bash
grasp-train ... --full_finetune --no-use_4bit
```

### Contract validation errors

Each contract row must have `id`, `instruction`, `input`, and `output`. The `output` field must be a valid JSON string with exactly one tool call.

Regenerate contract/chat files from raw generator JSONL:

```bash
python -m llm.prepare_llm_data \
  --generator_jsonl path/to/grasp_gen.jsonl \
  --out_contract path/to/llm_contract.jsonl \
  --out_chat path/to/llm_chat.jsonl
```

### Model emits extra text

Dump generations and inspect failures:

```bash
grasp-eval \
  --contract_jsonl data/ycb/01/llm_contract_10000.jsonl \
  --model_path models/qwen2_5_3b_ycb_lora \
  --max_examples 100 \
  --dump_all_jsonl eval_outputs/debug_generations.jsonl
```

Then consider more data from the failing context, a lower learning rate, or a shorter `max_seq_length` to reduce truncation pressure.
