# grasp-copilot
a copilot llm for suggesting grasp assistives.

For a complete end-to-end guide to data generation, dataset preparation, model training, evaluation, inference, and model storage, see [`README_DATA_TRAINING.md`](README_DATA_TRAINING.md).

## Installation

This project assumes a Conda environment named **`llm`** (no venv).

### Create the Conda env

```bash
conda create -n llm python=3.11 -y
conda activate llm
```

### Install PyTorch (pick ONE)

CPU-only:

```bash
conda install -y -c pytorch pytorch cpuonly
```

NVIDIA CUDA (example for H100; pick the CUDA version that matches your stack):

```bash
conda install -y -c pytorch -c nvidia pytorch pytorch-cuda=12.1
```

### Install grasp-copilot + Python deps

Recommended (editable install, enables `python -m data_generator...` and `grasp-*` CLIs):

```bash
python -m pip install -e grasp-copilot
```

Optional extras:

```bash
python -m pip install -e "grasp-copilot[test]"   # pytest
python -m pip install -e "grasp-copilot[qlora]"  # bitsandbytes for --use_4bit
```

Alternate (pip requirements file):

```bash
python -m pip install -r grasp-copilot/requirements.txt
```

### One-command installer (optional)

If you want an end-to-end installer that creates the env, installs PyTorch, and installs the package:

```bash
bash grasp-copilot/install_conda.sh --cuda 12.1
conda activate llm
```

## Dataset generation (collect + prepare)

Use the one-shot command below. It will:
- Generate the raw generator JSONL
- Convert it into **contract JSONL** (for training/eval)
- Convert contract → **chat JSONL** (for some trainers)

```bash
conda activate llm
```

### Collect + prepare (default output dir is auto-numbered)

```bash
conda activate llm
grasp-collect --episodes 1000 --seed 0
```

If you prefer Python module form:

```bash
conda activate llm
python -m data_generator.collect_and_prepare --episodes 1000 --seed 0
```

By default, this allocates a new run directory under:

- `grasp-copilot/data/runs/001`
- `grasp-copilot/data/runs/002`
- ...

Each run directory contains:
- `grasp_gen.jsonl` (+ `grasp_gen.jsonl.stats.json`)
- `llm_contract.jsonl`
- `llm_chat.jsonl`

### Balancing / rebalancing (recommended for training)

The dataset is naturally **INTERACT-heavy**. Rebalancing helps the model learn to emit motion tools when appropriate.

```bash
conda activate llm
grasp-collect --episodes 10000 --rebalance
```

This writes additional files:
- `llm_contract_rebalanced.jsonl`
- `llm_chat_rebalanced.jsonl`

Custom knobs:

```bash
grasp-collect --episodes 10000 \
  --motion_repeat 10 \
  --interact_keep_prob 0.7 \
  --rebalance_seed 0
```

### Data generation flags (most useful)

- **`--episodes`**: number of scripted episodes to generate
- **`--seed`**: RNG seed
- **`--n_obj_min` / `--n_obj_max`**: number of objects in the scene
- **`--collision_p`**: collision probability used in scene sampling
- **`--candidate_max_dist`**: candidate generation radius
- **`--skip_prepare`**: only write `grasp_gen.jsonl` (no contract/chat)
- **`--generator_jsonl`**: skip collection and re-prepare from an existing `grasp_gen.jsonl`
- **Balancing**: `--rebalance` or (`--motion_repeat`, `--interact_keep_prob`, `--rebalance_seed`)

## Training

Training consumes **contract JSONL** (typically `llm_contract_rebalanced.jsonl`).

### Full fine-tune (big VRAM GPUs; e.g. H100)

```bash
conda activate llm
python grasp-copilot/scripts/train_sft_lora.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --train_path grasp-copilot/data/runs/001/llm_contract_rebalanced.jsonl \
  --valid_path grasp-copilot/data/runs/001/llm_contract.jsonl \
  --full_finetune \
  --no-use_4bit \
  --max_seq_length 2048 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --lr 2e-5 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --save_strategy steps \
  --save_steps 500 \
  --eval_steps 500
```

Notes:
- Checkpoints are written under `"<output_dir>_adapter/checkpoint-*/"` (unless `--save_strategy no`).
- `--save_steps` controls checkpoint cadence; `--eval_steps` controls eval cadence.

### LoRA / QLoRA (smaller VRAM GPUs)

QLoRA is the default (`--use_4bit` defaults to true). Install bitsandbytes if needed:

```bash
python -m pip install -e "grasp-copilot[qlora]"
```

Example:

```bash
conda activate llm
python grasp-copilot/scripts/train_sft_lora.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --train_path grasp-copilot/data/runs/001/llm_contract_rebalanced.jsonl \
  --valid_path grasp-copilot/data/runs/001/llm_contract.jsonl \
  --max_seq_length 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --lr 2e-4 \
  --optim paged_adamw_32bit \
  --logging_steps 20 \
  --save_strategy no
```

If you hit CUDA OOM:
- Reduce `--max_seq_length` (e.g. 512)
- Disable eval: `--disable_eval`

## Evaluation

Evaluate a merged model against a contract JSONL and dump mistakes:

```bash
conda activate llm
grasp-eval \
  --contract_jsonl grasp-copilot/data/runs/001/llm_contract.jsonl \
  --model_path grasp-copilot/models/qwen2_5_3b_instruct_ft_001 \
  --max_examples 500 \
  --dump_mistakes_jsonl grasp-copilot/eval_outputs/eval_001_mistakes.jsonl
```

## Hugging Face model storage

The model directories in `/media/ali/USB/old` are merged Hugging Face model
directories when they contain `config.json`, tokenizer files, and
`model*.safetensors`. Upload them to private Hugging Face Hub repos:

```bash
conda activate llm
hf auth login
# Older installs also support:
# huggingface-cli login
python grasp-copilot/scripts/hf_model_store.py list --source-root /media/ali/USB/old
python grasp-copilot/scripts/hf_model_store.py upload \
  --source-root /media/ali/USB/old \
  --namespace YOUR_HF_USERNAME_OR_ORG
```

The current uploaded repos are:

- `alirb97/grasp-copilot-qwen2_5_3b_oracle_lora`
- `alirb97/grasp-copilot-qwen2_5_3b_oracle_lora_pouring`
- `alirb97/grasp-copilot-qwen2_5_3b_oracle_lora_stacking`
- `alirb97/grasp-copilot-qwen2_5_3b_oracle_lora_ycb`
- `alirb97/grasp-copilot-qwen2_5_3b_oracle_woz_lora`
- `alirb97/grasp-copilot-qwen2_5_3b_oracle_woz_lora_r32`

These repos are private unless they are changed to public on Hugging Face. A
different machine or collaborator can load them if they first authenticate with
a Hugging Face token that has access:

```bash
conda activate llm
hf auth login
# Older installs also support:
# huggingface-cli login
```

Upload only one model:

```bash
python grasp-copilot/scripts/hf_model_store.py upload \
  --source-root /media/ali/USB/old \
  --model qwen2_5_3b_oracle_woz_lora \
  --repo-id YOUR_HF_USERNAME_OR_ORG/grasp-copilot-qwen2_5_3b_oracle_woz_lora
```

Retrieve it later into this repo:

```bash
python grasp-copilot/scripts/hf_model_store.py download \
  --repo-id YOUR_HF_USERNAME_OR_ORG/grasp-copilot-qwen2_5_3b_oracle_woz_lora \
  --target-dir grasp-copilot/models \
  --local-name qwen2_5_3b_oracle_woz_lora
```

You can also load a Hub repo id directly anywhere this project accepts
`--model_path`:

```bash
grasp-infer \
  --model_path alirb97/grasp-copilot-qwen2_5_3b_oracle_woz_lora \
  --prompt 'Return {"tool":"INTERACT","args":{"kind":"QUESTION","text":"ok?","choices":["yes","no"]}}'
```

The first full inference run downloads the model weights, about 5.8 GB per
repo, into the Hugging Face cache.

Lightweight verification without downloading all weight shards:

```bash
python - <<'PY'
from transformers import AutoConfig, AutoTokenizer

repo = "alirb97/grasp-copilot-qwen2_5_3b_oracle_woz_lora"
cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True, use_fast=True)
print(cfg.model_type, tok.__class__.__name__, len(tok))
PY
```

## Demos

### JSON-only inference (CLI)

```bash
conda activate llm
grasp-infer --model_path grasp-copilot/models/qwen2_5_3b_instruct_ft_001 --prompt 'Return {"tool":"INTERACT","args":{"kind":"QUESTION","text":"ok?","choices":["yes","no"]}}'
```

### GUI demo (oracle or HF model)

Oracle backend:

```bash
conda activate llm
python grasp-copilot/scripts/gui_assist_demo.py --backend oracle
```

HF backend:

```bash
conda activate llm
python grasp-copilot/scripts/gui_assist_demo.py --backend hf --model_path grasp-copilot/models/qwen2_5_3b_instruct_ft_001
```
