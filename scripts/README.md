# `scripts/` — Oracle demos and supporting CLIs

End-user CLIs for running the oracle visually, inspecting data, doing
inference, and training. The headline file is `gui_assist_demo.py`,
which lets you exercise any of the three oracles (or a fine-tuned
model) in an interactive Tk window.

---

## 1 · `gui_assist_demo.py` — visual oracle/model tester

A single window with:

- A 3×3 grid canvas drawing the workspace, objects, and gripper (yaw
  arrow + z indicator).
- Per-env extras: **stacking** highlights the held cube and shades
  covered bases; **pouring** draws pitchers/cups with fill-level bars
  (EMPTY/PARTIAL/FULL).
- A right-side log of the dialog (assistant prompts + your replies).
- A mode panel: `1` translation · `2` rotation · `3` gripper.
- An **"Ask assistance"** button that queries the oracle (or HF model)
  for the next `{tool, args}` JSON output.
- INTERACT prompts render as discrete buttons; clicking one auto-applies
  the next oracle step.
- A **Reset** button to resample a new scene.

### Run the oracle for each env

```bash
conda activate llm

# YCB reach-to-grasp (default; same as before)
python scripts/gui_assist_demo.py --backend oracle

# Cube stacking
python scripts/gui_assist_demo.py --env cube_stacking --backend oracle

# Pouring (3 buckets: SMALL / HALF / FULL)
python scripts/gui_assist_demo.py --env pouring --backend oracle
```

### Run a fine-tuned model instead of the oracle

`--backend hf` loads a **merged** HuggingFace model and queries it for
each "Ask assistance" press. The model sees the same JSON input the
training data was built from, so this is the closest-to-prod check.

```bash
# Trained on YCB
python scripts/gui_assist_demo.py --backend hf \
    --model_path models/qwen2_5_7b_instruct_ft

# Trained on stacking
python scripts/gui_assist_demo.py --env cube_stacking --backend hf \
    --model_path models/qwen2_5_7b_instruct_stacking

# Trained on pouring
python scripts/gui_assist_demo.py --env pouring --backend hf \
    --model_path models/qwen2_5_7b_instruct_pouring
```

Pass `--deterministic` to force greedy decoding for stable debugging.

### Keyboard controls

| Keys | What they do |
|---|---|
| `1` / `2` / `3` | Switch input mode (translation / rotation / gripper). |
| `Arrow keys` | **Translation mode:** move gripper one cell. **Rotation mode:** rotate yaw (left = CCW, right = CW). **Gripper mode:** open (up) / close (down). |
| `Q` / `E` | Rotate yaw CCW / CW (also switches to rotation mode). |
| `W` / `S` | Move z up / down (also switches to gripper mode). |

### Useful flags

| Flag | Default | Notes |
|---|---|---|
| `--env {reach_to_grasp_ycb, cube_stacking, pouring}` | `reach_to_grasp_ycb` | Pick the oracle / scene generator. |
| `--backend {oracle, hf}` | `oracle` | `hf` loads a model via `--model_path`. |
| `--n_obj N` | env-aware (ycb=8, stacking=4, pouring=3) | Number of objects to spawn. |
| `--candidate_max_dist N` | env-aware (ycb=1, stacking=2, pouring=2) | Manhattan radius for the candidate set. |
| `--collision_p P` | `0.2` | YCB only: probability two objects share a cell. |
| `--seed N` | `0` | Scene RNG seed. |
| `--model_path PATH` | — | HF backend: path or HF Hub id of a **merged** model. |
| `--temperature T` / `--top_p P` / `--max_new_tokens N` | `0.2 / 0.9 / 256` | HF sampling knobs. |
| `--deterministic` | off | HF backend: greedy decoding + deterministic torch settings. |
| `--use_4bit` | off | HF backend: load in 4-bit (requires `bitsandbytes`). |

### Tips when testing the oracles

- Watch the log: every assistant turn shows up as `[assistant] {…JSON…}`
  and every user reply as `[user] …`. If you see two motion tools in a
  row without a CONFIRM in between, that's a bug — file it.
- Press **Reset** between runs to resample the scene without restarting
  the process.
- For **pouring**, drive the gripper toward the pitcher first; the
  oracle should suggest grabbing it (`pitcher_acquisition`). After a
  `GRAB`, head toward a non-full cup and the amount sub-flow
  (`amount_choice` → `confirm_amount` → `POUR`) should fire.
- For **stacking**, the episode always starts with one cube already
  held. Try aiming at a base cube that already has another on top —
  the oracle should emit `non_top_redirect`.

---

## 2 · `inference_demo.py` — headless model inference

Run a fine-tuned model on a single JSON input from the command line.
Useful for batch eval and shell pipelines.

```bash
# Trained YCB model
python scripts/inference_demo.py \
    --model_path models/qwen2_5_7b_instruct_ft \
    --prompt '{"objects":[...],"gripper_hist":[...],"memory":{...},"user_state":{"mode":"translation"}}'

# Or feed a JSONL of inputs and dump tool calls
python scripts/inference_demo.py \
    --model_path models/qwen2_5_7b_instruct_ft \
    --input_jsonl data/<env>/01/llm_contract_<N>.jsonl \
    --output_jsonl predictions.jsonl
```

Backed by `llm/inference.py`.

---

## 3 · `inspect_data.py` — peek inside generated JSONL

```bash
# Raw oracle output (per-tick records)
python scripts/inspect_data.py --file data/<env>/01/grasp_gen_<N>.jsonl --mode generator --n 5

# SFT contract (instruction/input/output)
python scripts/inspect_data.py --file data/<env>/01/llm_contract_<N>.jsonl --mode contract --n 5

# Qwen chat format (messages)
python scripts/inspect_data.py --file data/<env>/01/llm_chat_<N>.jsonl --mode chat --n 5
```

---

## 4 · `prepare_llm_data.py` — convert generator → contract / chat

Thin shim around `llm.prepare_llm_data`. Prefer the module form when
scripting:

```bash
python -m llm.prepare_llm_data \
    --generator_jsonl data/<env>/01/grasp_gen_<N>.jsonl \
    --out_contract    data/<env>/01/llm_contract_<N>.jsonl \
    --out_chat        data/<env>/01/llm_chat_<N>.jsonl
```

For most users, `grasp-collect` already does this in one shot — only
reach for `prepare_llm_data` if you have an existing generator JSONL
to re-prepare.

---

## 5 · `rebalance_contract.py` — upsample motion-tool examples

If you already have an `llm_contract.jsonl` and want a motion-heavier
variant for training:

```bash
python scripts/rebalance_contract.py \
    --in_path    data/<env>/01/llm_contract_<N>.jsonl \
    --out_path   data/<env>/01/llm_contract_<N>_rebalanced.jsonl \
    --motion_repeat 10 \
    --interact_keep_prob 0.7 \
    --seed 0
```

`grasp-collect --rebalance` invokes this automatically, so you only
need it for ad-hoc resampling.

---

## 6 · `train_sft_lora.py` — SFT / LoRA / QLoRA fine-tuning

Full-parameter fine-tune on H100-class GPUs:

```bash
python scripts/train_sft_lora.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --train_path data/<env>/01/llm_contract_<N>_rebalanced.jsonl \
    --valid_path data/<env>/01/llm_contract_<N>.jsonl \
    --full_finetune --no-use_4bit \
    --max_seq_length 2048 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr 2e-5 --num_train_epochs 1 \
    --logging_steps 10 --save_strategy steps --save_steps 500 --eval_steps 500 \
    --output_dir models/qwen2_5_7b_instruct_<env>
```

QLoRA on a smaller GPU:

```bash
python scripts/train_sft_lora.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --train_path data/<env>/01/llm_contract_<N>_rebalanced.jsonl \
    --valid_path data/<env>/01/llm_contract_<N>.jsonl \
    --max_seq_length 1024 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
    --lr 2e-4 --optim paged_adamw_32bit \
    --logging_steps 20 --save_strategy no
```

See `grasp-copilot/README.md` for the full training section.

---

## 7 · `merge_lora.py` *(deprecated)*

Older shim for merging LoRA adapters into a standalone model. New
training runs write merged checkpoints directly, so you generally
don't need this. Kept around so legacy adapter dirs can still be
flattened:

```bash
python scripts/merge_lora.py \
    --base_model_name Qwen/Qwen2.5-7B-Instruct \
    --adapter_dir   adapters/foo \
    --output_dir    models/foo_merged
```

---

## 8 · `_bootstrap.py`

Not a CLI — it's a path-fixup module imported at the top of every
script. Adds the repo root to `sys.path` so the scripts can be run
from any working directory.

---

## Cheat sheet

```bash
# 1) Visually exercise each oracle
python scripts/gui_assist_demo.py --env reach_to_grasp_ycb
python scripts/gui_assist_demo.py --env cube_stacking
python scripts/gui_assist_demo.py --env pouring

# 2) Generate datasets
grasp-collect --env reach_to_grasp_ycb --episodes 10000 --rebalance
grasp-collect --env cube_stacking      --episodes 10000 --rebalance
grasp-collect --env pouring            --episodes 10000 --rebalance

# 3) Inspect a sample
python scripts/inspect_data.py --file data/<env>/01/grasp_gen_<N>.jsonl --mode generator --n 5

# 4) Train
python scripts/train_sft_lora.py --train_path … --valid_path … --output_dir models/…

# 5) Visually evaluate the trained model
python scripts/gui_assist_demo.py --env <env> --backend hf --model_path models/<your_model>
```
