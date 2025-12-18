# grasp-copilot
a copilot llm for suggesting grasp assistives.

## Dataset generation + inspection

All commands must activate the `talm` environment first:

```bash
conda activate talm
```

### Generate a dataset (raw generator JSONL)

```bash
conda activate talm
python -m data_generator.generate_dataset --episodes 1000 --seed 0 --out /tmp/grasp_gen.jsonl
# also writes: /tmp/grasp_gen.jsonl.stats.json
```

### Inspect the raw generator dataset (recommended for `/tmp/grasp_gen.jsonl`)

The generator records contain keys like `episode_id`, `t`, `objects`, `gripper_hist`, `memory`, `target_tool_call`.

```bash
conda activate talm
python -m data_generator.inspect_data --path /tmp/grasp_gen.jsonl --summary
python -m data_generator.inspect_data --path /tmp/grasp_gen.jsonl --episode 0 --max-t 20 --show-objects --show-gripper --show-memory
```

### Convert generator JSONL → LLM contract/chat JSONL, then inspect

```bash
conda activate talm
python scripts/prepare_llm_data.py --generator_jsonl /tmp/grasp_gen.jsonl --out_contract /tmp/grasp_contract.jsonl --out_chat /tmp/grasp_chat.jsonl

python scripts/inspect_data.py --file /tmp/grasp_contract.jsonl --mode contract --n 3
python scripts/inspect_data.py --file /tmp/grasp_chat.jsonl --mode chat --n 1
```

Note: `python scripts/inspect_data.py --mode generator` expects different fields (`obs`, `dialog`) than the raw generator output; for raw generator inspection use `python -m data_generator.inspect_data`.

## LLM fine-tuning + inference

All commands must activate the `talm` environment first:

```bash
conda activate talm
```

### Prepare LLM data (adapter over existing generator JSONL)

```bash
conda activate talm
python scripts/prepare_llm_data.py --generator_jsonl data.jsonl --out_contract llm_contract.jsonl --out_chat llm_chat.jsonl
```

### Train SFT + LoRA / QLoRA

```bash
conda activate talm
python scripts/train_sft_lora.py --train_path llm_contract.jsonl --valid_path llm_contract.jsonl --output_dir outputs/qwen_lora --model_name Qwen/Qwen2.5-7B-Instruct --use_4bit
```

### Merge LoRA (optional)

```bash
conda activate talm
python scripts/merge_lora.py --base_model_name Qwen/Qwen2.5-7B-Instruct --adapter_dir outputs/qwen_lora --output_dir outputs/qwen_merged
```

### Inference demo (JSON-only)

```bash
conda activate talm
python scripts/inference_demo.py --model_name Qwen/Qwen2.5-7B-Instruct --adapter_path outputs/qwen_lora --prompt 'Return {"tool_name":"INTERACT","arguments":{"type":"notify","text":"ok"}}'
```
