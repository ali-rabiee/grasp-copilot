# `data_generator/` — Oracle-driven data collection

Scripted oracles that emit PRIME-schema (`{tool, args}`) supervision for
three environments:

| `--env` | Module | Skills the oracle can emit |
|---|---|---|
| `reach_to_grasp_ycb` (default) | `oracle.py` + `episode.py` | `INTERACT`, `APPROACH`, `ALIGN_YAW` |
| `cube_stacking` | `oracle_stacking.py` + `episode_stacking.py` | `INTERACT`, `APPROACH`, `ALIGN_YAW`, `STACK`, `RELEASE` |
| `pouring` | `oracle_pouring.py` + `episode_pouring.py` | `INTERACT`, `APPROACH`, `ALIGN_YAW`, `GRAB`, `POUR(amount ∈ {SMALL,HALF,FULL})` |

Each oracle shares the same `OracleState` machinery
(`oracle.py`) — same awaiting-cascade, same `validate_tool_call`,
same `_rank_candidates`. The new oracles add their own prompt types
and **always emit a CONFIRM (or YES/NO SUGGESTION) before any motion
tool fires** — there is no silent escalation.

The env dispatcher lives in `oracle_registry.py`:

```python
from data_generator.oracle_registry import get_spec
spec = get_spec("pouring")
spec.decide_fn(objects, gripper_hist, memory, state, user_state=us)
```

---

## 1 · One-shot collection (recommended)

The user-facing entry point is `grasp-collect`
(installed as a console script by `pyproject.toml`) or the equivalent
module form `python -m data_generator.collect_and_prepare`.

### YCB (default — unchanged from before)

```bash
conda activate llm
grasp-collect --episodes 10000 --seed 0 --rebalance
```

### Cube stacking

```bash
grasp-collect --env cube_stacking --episodes 10000 --seed 0 --rebalance
```

### Pouring

```bash
grasp-collect --env pouring --episodes 10000 --seed 0 --rebalance
```

Each run writes into a fresh `data/runs/NNN/` directory (auto-numbered):

```
data/runs/NNN/
├── grasp_gen.jsonl                  ← raw per-tick generator output (this file is the "ground truth")
├── grasp_gen.jsonl.stats.json       ← tool distribution, user-reply distribution
├── llm_contract.jsonl               ← {instruction, input, output} for SFT
├── llm_contract_rebalanced.jsonl    ← motion-tool-upsampled training file (if --rebalance)
├── llm_chat.jsonl                   ← Qwen chat format
└── llm_chat_rebalanced.jsonl
```

`--rebalance` upsamples motion-tool examples (`APPROACH/ALIGN_YAW/STACK/GRAB/POUR`)
because the natural distribution is INTERACT-heavy (~85–90 %).
You almost always want this for training.

---

## 2 · Important arguments

| Flag | Default | Notes |
|---|---|---|
| `--episodes N` | **10000** | Number of episodes to roll out. |
| `--env NAME` | `reach_to_grasp_ycb` | Pick the oracle/env. |
| `--seed N` | `0` | Reproducibility. Use a different seed per parallel run. |
| `--out_dir PATH` | auto-numbered | If set, writes there instead of `data/runs/NNN/`. |
| `--n_obj_min / --n_obj_max` | `2 / 10` | YCB / stacking scene size range. Pouring ignores these (uses 1–3 cups). |
| `--collision_p` | `0.15` | YCB only: prob two objects share a cell. |
| `--candidate_max_dist` | env default | Manhattan radius for the candidate set. Defaults: ycb=1, stacking=2, pouring=2. |
| `--rebalance` | off | Convenience: enable motion-tool upsampling with reasonable defaults. |
| `--motion_repeat N` | `1` | Repeat each motion-tool example N times in the contract. |
| `--interact_keep_prob P` | `1.0` | Probability of keeping each INTERACT example. |
| `--skip_prepare` | off | Only write `grasp_gen.jsonl`; skip contract/chat conversion. |
| `--generator_jsonl PATH` | — | Skip collection; re-prepare an existing generator JSONL. |

The full list is available via `grasp-collect --help`.

---

## 3 · Collecting all three envs for a single training run

For a unified model that handles all task families, generate each env
into its own subfolder, then either:

**(a)** train on a merged contract:

```bash
# generate
grasp-collect --env reach_to_grasp_ycb --episodes 10000 --seed 0 --rebalance \
              --out_dir data/runs/merged_v1/ycb
grasp-collect --env cube_stacking      --episodes 10000 --seed 0 --rebalance \
              --out_dir data/runs/merged_v1/stacking
grasp-collect --env pouring            --episodes 10000 --seed 0 --rebalance \
              --out_dir data/runs/merged_v1/pouring

# concat (the contract format is identical across envs)
cat data/runs/merged_v1/{ycb,stacking,pouring}/llm_contract_rebalanced.jsonl \
    > data/runs/merged_v1/llm_contract_rebalanced.jsonl

# convert to chat format for the trainer
python -m llm.data convert-contract-to-chat \
    --in data/runs/merged_v1/llm_contract_rebalanced.jsonl \
    --out data/runs/merged_v1/llm_chat_rebalanced.jsonl
```

**(b)** or train per-env adapters separately and merge later.

---

## 4 · Lower-level entry: `python -m data_generator.generate_dataset`

Skips the contract/chat conversion — useful for debugging the oracle:

```bash
python -m data_generator.generate_dataset \
    --env cube_stacking --episodes 100 --seed 0 \
    --out /tmp/stacking_check.jsonl
```

Outputs:

- `/tmp/stacking_check.jsonl`
- `/tmp/stacking_check.jsonl.stats.json` (tool distribution + user-reply distribution per context)

---

## 5 · Inspecting and validating generated data

Quick eyeball:

```bash
python scripts/inspect_data.py --file data/runs/NNN/grasp_gen.jsonl --mode generator --n 5
python scripts/inspect_data.py --file data/runs/NNN/llm_contract.jsonl --mode contract --n 5
python scripts/inspect_data.py --file data/runs/NNN/llm_chat.jsonl     --mode chat     --n 5
```

Programmatic sanity (every record schema-valid, motion always preceded
by consent):

```python
import json
from data_generator.oracle import validate_tool_call
for env in ("reach_to_grasp_ycb", "cube_stacking", "pouring"):
    path = f"data/runs/NNN/grasp_gen.jsonl"   # one path per env
    for line in open(path):
        rec = json.loads(line)
        validate_tool_call(rec["target_tool_call"], env=rec.get("env", env))
```

Stats file gives the tool distribution and per-context reply counts:

```bash
cat data/runs/NNN/grasp_gen.jsonl.stats.json | jq .
```

---

## 6 · How the per-env oracles differ from YCB

| Oracle | New prompt types | New tools | Special sub-flow |
|---|---|---|---|
| `oracle_stacking` | `intent_gate_stack`, `confirm_stack`, `non_top_redirect` | `STACK(obj)`, `RELEASE` | Detects "this base already has a cube on top" → suggests an alternative. |
| `oracle_pouring`  | `pitcher_acquisition`, `intent_gate_pour`, `confirm_pour`, `confirm_grab`, `amount_choice`, `confirm_amount`, `cup_full_redirect` | `GRAB(obj)`, `POUR(obj, amount)` | Three-step "what / where / how much": confirms target, asks the amount bucket, then confirms the chosen amount before firing `POUR`. |

Both oracles share YCB's awaiting cascade, candidate ranking,
oscillation detectors, episode-start intent gate, and `anything_else`
recovery.

---

## 7 · Testing the oracles visually

See `scripts/README.md` — `gui_assist_demo.py` accepts `--env`
and runs any of the three oracles as a Tk keyboard-driven simulator.

---

## 8 · File layout

```
data_generator/
├── README.md                   (this file)
├── grid.py                     3×3 cell math (Manhattan, step_toward, neighbors)
├── yaw.py                      8-bin yaw arithmetic (move_toward, neighbors, cyclic distance)
├── oracle.py                   YCB oracle + shared OracleState/validate_tool_call/ENV_SKILLS/POUR_AMOUNTS
├── oracle_stacking.py          cube_stacking decision tree
├── oracle_pouring.py           pouring decision tree (with amount sub-flow)
├── oracle_registry.py          env_name → (Episode class, decide fn, candidate radius)
├── episode.py                  YCB Episode (scene sampler + user-motion sim)
├── episode_stacking.py         EpisodeStacking (held cube, pre-stack, top_of_stack)
├── episode_pouring.py          EpisodePouring (pitcher kind, fill levels)
├── generate_dataset.py         Env-aware generate() + user-reply simulator
├── collect_and_prepare.py      grasp-collect CLI (collection + contract/chat conversion)
├── inspect_data.py             Library used by scripts/inspect_data.py
├── run_dirs.py                 Numbered output dir allocator
└── tests/                      Pytest cases for the YCB oracle's invariants
```

---

## 9 · Relationship to the wizard pipeline

The Wizard-of-Oz package at `grasp-copilot/wizard/` is an alternative
supervision source — a human in the loop instead of the scripted
oracle. The output JSONL schema is identical, so any wizard run can
be fed through `--generator_jsonl` to `grasp-collect` for contract
preparation. See `wizard/README.md`.

For paper experiments the oracle is the **fast iteration** path
(seconds per 1k episodes); wizard data is the headline
supervision-quality path.
