# Package 1 — Wizard-of-Oz (WoZ) Data Collection and Training

This package replaces the heuristic **oracle** (`grasp-copilot/data_generator/oracle.py`)
as PRIME's training-time teacher with a **human wizard** that sees the *exact* same
symbolic observation the deployed LLM will see, and emits the *exact* same JSON
tool-call schema. The result is a supervised dataset that captures expert
ask-or-act judgement on ambiguous interaction states the rule-based oracle
demonstrably mishandles.

---

## 1 · Why WoZ instead of (or in addition to) the oracle

The oracle in `data_generator/oracle.py` encodes hand-coded rules: "ask if N≥2
candidates," "fire `ALIGN_YAW` if yaw oscillation triggers," etc. It is brittle
under noisy gripper history — Table II in the paper shows the oracle losing
12 pp tool accuracy at p=0.5 input noise, while the fine-tuned 7B model loses
only 5.1 pp. WoZ lets a human apply *contextual* judgement on exactly the same
inputs and supervises the LLM to imitate that judgement.

WoZ is the new headline supervision; oracle data is retained as an **ablation**.

---

## 2 · The "wizard sees what the LLM sees" rule (non-negotiable)

The wizard interface presents **only** the symbolic state `s_t = (O, H_t, m_t)`
plus the structured memory `M_t`. It does **not** show:

- A photorealistic 3D rendering of the scene
- The ground-truth user intent (target object id)
- Object metric coordinates beyond the 3×3 grid cell + 8-bin yaw
- Anything not available to the deployed LLM at inference time

If the wizard sees more than the LLM, the resulting policy is unbounded above
by the model's observation space and the dataset is undeployable. Treat this
as a hard invariant of the codebase.

What the wizard *does* see (full list, mirroring `extractor.InputExtractor`):

| Field | Shape / type | Example |
|---|---|---|
| `objects` | list of `{id, label, cell, yaw, is_held}` | `[{id:"obj0", label:"banana", cell:"B2", yaw:"NE", is_held:false}, ...]` |
| `gripper_hist` | deque of last 6 `{cell, yaw, z}` records | `[{cell:"A2", yaw:"N", z:"high"}, ...]` |
| `user_state.mode` | one of `translation` / `rotation` / `gripper` | `"translation"` |
| `memory.candidates` | list of plausible obj ids | `["obj0", "obj3"]` |
| `memory.past_dialogs` | ordered prompt/option/response turns | `[{prompt:..., choices:[...], reply:"YES"}]` |
| `memory.last_action` | most recent skill call + outcome | `{tool:"APPROACH", obj:"obj3", outcome:"success"}` |
| `memory.excluded_obj_ids` | objects pruned by user replies | `["obj1"]` |
| `memory.last_tool_calls` | rolling history (last ~3 calls) | `["INTERACT", "APPROACH"]` |

---

## 3 · Schematic GUI design

A full IsaacSim render is **not** required. The wizard interface is a
lightweight 2D schematic so the wizard can read the symbolic state at a
glance. Suggested layout (single window, ~1100 × 700 px):

```
┌────────────────────────────────────────────────────────────────────────┐
│  PRIME Wizard · Episode 042 · Tick 17 · Mode: TRANSLATION              │
├────────────────────────────────────┬───────────────────────────────────┤
│  Workspace 3×3 (top-down)          │  Memory                           │
│                                    │                                   │
│   A1 │ A2 │ A3                     │  Candidates:  [obj0, obj3]        │
│  ────┼────┼────                    │                                   │
│   B1 │ B2 │ B3      G̲ → cell:B2    │  Past dialogs:                    │
│  ────┼────┼────     yaw:NE  z:hi   │   1. "Trying obj3?" → NO          │
│   C1 │ C2 │ C3                     │   2. "Confirm obj0?" → (pending)  │
│                                    │                                   │
│   ◯ obj0 banana  cell:B2 yaw:N     │  Last action:                     │
│   ◯ obj1 mug     cell:C3 yaw:E     │   APPROACH(obj3) → fail (timeout) │
│   ◯ obj2 cup     cell:A1 yaw:S     │                                   │
│   ◯ obj3 can     cell:B2 yaw:NE    │  Excluded: [obj3]                 │
│                                    │                                   │
│   Gripper trail: A1→A2→B2→B2→B2→B2 │  Last 3 calls: [APPROACH,         │
│   (most recent rightmost)          │   INTERACT, INTERACT]             │
├────────────────────────────────────┴───────────────────────────────────┤
│  ⚠  WIZARD ALERT — make a decision                                     │
│                                                                        │
│  ┌─ Interaction ─────────────────────┐  ┌─ Execution ─────────────────┐│
│  │ kind:  ( ) QUESTION               │  │ tool:                       ││
│  │        ( ) CONFIRM                │  │   ( ) APPROACH(obj0)        ││
│  │        ( ) SUGGESTION             │  │   ( ) APPROACH(obj1)        ││
│  │ text:  [_______________________]  │  │   ( ) APPROACH(obj3)        ││
│  │ opts:  [1)____] [2)____] [3)____] │  │   ( ) ALIGN_GRIPPER(obj0)   ││
│  │        [4)____] [5)____]          │  │   ...                       ││
│  └───────────────────────────────────┘  └─────────────────────────────┘│
│                                                                        │
│  [ Submit ]    [ Skip / Defer to next tick ]                           │
└────────────────────────────────────────────────────────────────────────┘
```

Implementation notes:

- The grid is just a 3×3 matplotlib axis (or Tkinter canvas). Each
  object is a colored dot at its `(cell, yaw)`; yaw is rendered as a short
  line out of the dot in the 8-bin direction.
- The gripper trail is the same axis with a darker marker at the latest
  cell and faded markers/arrows for the previous 5 history entries.
- The "z" bin (`low`/`mid`/`high`) is a small badge next to the gripper —
  no need to fake 3D depth.
- Memory is plain text panels.
- Decision controls submit one JSON object. Validate against
  `data_generator.oracle.validate_tool_call` before writing to disk so the
  schema constraints (e.g. `MAX_INTERACT_CHOICES = 5`) are enforced
  identically to the oracle pipeline.

Recommended stack: **Tkinter or PyQt for the GUI + matplotlib for the grid
panel**. No Isaac Sim window. Anything that runs as a normal Python process
on the wizard's laptop is fine.

---

## 4 · Data collection protocol

### 4.1 Environments

Run the wizard episode driver against the existing Isaac Sim environments
(headless is fine — the wizard does *not* see the renderer):

| Env | Module | Purpose |
|---|---|---|
| YCB reach-to-grasp | `kinova-isaac/environments/ycb_reach_to_grasp/` | Original PRIME task family |
| Cube stacking | `kinova-isaac/DEMOS/block_stacking/` | Multi-step manipulation |
| Pouring | `kinova-isaac/environments/pouring/` | Container + target distinction |

Episode-level randomization (mirror `data_generator.collect_and_prepare`):

- 2–10 YCB objects per scene, drawn from the same YCB set the oracle uses.
- Random initial gripper pose within workspace bounds (workspace_min/max from
  `kinova-isaac/data_collection/profiles/vla_v1.py`).
- Random `user_state.mode` per episode (translation / rotation / gripper),
  and mid-episode mode switches with the same distribution as the oracle.
- Random simulated-user trajectory (see §4.2).

### 4.2 Simulated user (reused from Package 3)

A scripted "user model" drives the gripper toward a randomly selected target
object using noisy, low-bandwidth inputs (joystick-like deltas with
configurable jitter and direction-reversal probability). This produces the
state stream the wizard observes. The target id is recorded in the episode
metadata for offline analysis but is **never** displayed to the wizard.

The same user model from Package 3 is reused here so that the wizard
training distribution matches the real user-study distribution as closely as
possible.

### 4.3 Decision points (when the wizard is alerted)

The wizard does not annotate every tick. At each tick the driver decides
whether this is a *decision point*:

1. **Stochastic schedule.** With probability `p_alert` (default `0.15` per
   logging tick at `log_rate_hz=5`, ≈ one alert per ~1.3 s of sim time),
   pause the episode and surface the GUI alert.
2. **Always-alert events** (deterministic, on top of the schedule):
   - First tick after an episode reset.
   - Immediately after an `APPROACH` or `ALIGN_GRIPPER` execution returns
     (success or failure outcome).
   - When the candidate set changes size (new candidate enters or one is
     pruned by a previous reply).

This stochastic-plus-event schedule keeps wizard cognitive load manageable
(~10–25 decisions per episode) while ensuring high-information transitions
are always labelled.

When the wizard submits a decision, the driver applies it just like the
deployed model would: `INTERACT` calls feed back into `memory.past_dialogs`
on the simulated user's reply; `APPROACH` / `ALIGN_GRIPPER` calls dispatch
through `DEMOS/copilot_demo/copilot_demo/executor.py`.

### 4.4 Volume and wizard count

- **Target:** ~500 episodes total, balanced across the three task families
  (≈ 165 each).
- **Wizards:** at least 2 (preferably 3) independent annotators. Episodes
  are partitioned so each wizard contributes ~250 episodes from disjoint
  scenes; the assignment is logged per episode for inter-wizard variance
  analysis.
- **Held-out agreement subset:** an additional ~50 scenarios annotated by
  *all* wizards independently (same state stream, parallel sessions). These
  scenarios are used for the kappa analysis in §5 and excluded from
  training.

---

## 5 · Inter-wizard agreement

On the held-out 50-scenario subset, compute:

1. **Cohen's κ on ask-vs-act** — binary collapse of the tool field
   (`INTERACT` vs `{APPROACH, ALIGN_GRIPPER}`).
2. **Cohen's κ (or Fleiss' κ for 3+ wizards) on tool selection given act** —
   the multi-class agreement on which execution skill / which target object,
   conditional on both wizards choosing to act.
3. **Free-marginal κ** as a sensitivity check, since the marginal
   distribution of `INTERACT` vs act is highly imbalanced.

**Pass threshold:** κ ≥ 0.7 on ask-vs-act decisions (validation criterion).

Report these as a supporting result in the paper rebuttal/appendix; they
preempt "your wizards are inconsistent" critiques and are required to
defend WoZ as a teacher.

---

## 6 · Training

Identical hyperparameters to the existing oracle-trained 7B run (so any
delta is attributable to the supervision signal, not optimization
choices):

```bash
conda activate llm
python grasp-copilot/scripts/train_sft_lora.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --train_path grasp-copilot/wizard/data/runs/woz_001/llm_contract_rebalanced.jsonl \
  --valid_path grasp-copilot/wizard/data/runs/woz_001/llm_contract.jsonl \
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
  --eval_steps 500 \
  --output_dir grasp-copilot/models/qwen2_5_7b_instruct_woz
```

The WoZ JSONL passes through the same `llm.prepare_llm_data` →
`llm_contract.jsonl` → `llm_chat.jsonl` pipeline as oracle data. The only
difference is the supervision label.

---

## 7 · Implemented directory layout

```
grasp-copilot/wizard/
├── README.md                  (this file)
├── __init__.py
├── __main__.py                (so `python -m wizard ...` works)
├── cli.py                     (`collect` and `kappa` subcommands)
├── env/
│   ├── __init__.py
│   └── schematic_env.py       (sim-free 3×3 world; objects + gripper + memory)
├── driver/
│   ├── __init__.py
│   ├── user_model.py          (noisy joystick user, intent-aware INTERACT replies)
│   ├── alert_scheduler.py     (stochastic + episode-start + post-exec + cand-change)
│   └── episode_runner.py      (per-episode loop; calls back into the wizard for decisions)
├── gui/
│   ├── __init__.py
│   ├── grid_view.py           (matplotlib 3×3 panel; objects, yaw arrows, gripper trail)
│   └── app.py                 (Tk window: header, grid, memory panel, decision form)
├── io/
│   ├── __init__.py
│   └── writer.py              (validates with `data_generator.oracle.validate_tool_call`,
│                                writes `grasp_gen.jsonl` + sidecar `episodes_meta.jsonl`)
├── analysis/
│   ├── __init__.py
│   └── kappa.py               (Cohen's κ, Fleiss' κ; ask-vs-act + tool-selection)
└── data/
    └── runs/                  (created at first run; not committed)
        └── woz_001/<wizard_id>/{grasp_gen.jsonl, episodes_meta.jsonl}
```

---

## 8 · Quick start (developer)

The package ships with a Tk + matplotlib GUI and a sim-free schematic
environment — **no Isaac Sim, no GPU, no Nucleus assets required for data
collection**. A wizard can run the entire pipeline from a laptop.

```bash
# 0. (One-time) editable install if not already done
python -m pip install -e grasp-copilot
conda activate llm

# 1. Launch the GUI for one wizard, one task family
python -m wizard collect \
    --env reach_to_grasp_ycb \
    --num-episodes 50 \
    --wizard-id alice \
    --p-alert 0.15 \
    --output grasp-copilot/wizard/data/runs/woz_001/alice
# (Or use the entry point: `grasp-wizard collect ...`)

# 2. Repeat for each env and each wizard, writing into separate sub-dirs:
#      .../woz_001/alice  .../woz_001/bob  .../woz_001/carol
#    For the held-out 50-scenario agreement subset, give every wizard the
#    same --seed and write into .../woz_001/agreement/<wizard_id>.

# 3. Compute inter-wizard agreement
python -m wizard kappa \
    --agreement-dir grasp-copilot/wizard/data/runs/woz_001/agreement \
    --out grasp-copilot/wizard/data/runs/woz_001/agreement/kappa_report.txt

# 4. Convert wizard JSONL → LLM training contract (existing pipeline)
python -m data_generator.collect_and_prepare \
    --generator_jsonl grasp-copilot/wizard/data/runs/woz_001/alice/grasp_gen.jsonl \
    --skip_collect

# 5. Fine-tune (see §6 — same hyperparameters as the oracle 7B run)
```

The collect subcommand:
- Runs the schematic episode loop in a worker thread.
- Pauses on every alert (per §4.3) and surfaces the GUI for the wizard to
  submit one validated tool call.
- Writes two files per run dir:
  - `grasp_gen.jsonl` — schema identical to oracle output, ready for the
    `llm.prepare_llm_data` pipeline. Contains **no** ground-truth fields.
  - `episodes_meta.jsonl` — sidecar with `intended_obj_id`, alert reasons,
    timings. Used by `wizard kappa` and by the ambiguous-scenario eval.
    Never feeds training.

---

## 9 · Deliverables

| # | Deliverable | Location |
|---|---|---|
| 1 | WoZ dataset (~500 episodes, JSON in PRIME schema) | `grasp-copilot/wizard/data/runs/woz_001/` |
| 2 | Fine-tuned WoZ-Qwen2.5-7B checkpoint | `grasp-copilot/models/qwen2_5_7b_instruct_woz/` |
| 3 | Inter-wizard agreement statistics | `wizard/data/runs/woz_001/agreement/kappa_report.txt` |
| 4 | Updated paper Table I with WoZ-7B as headline + Oracle-7B as ablation | `P_Rabiee_Copilot_IROS_2026/root.tex` |

---

## 10 · Validation criteria

The package is "done" only when **all three** are satisfied:

1. **WoZ-7B tool accuracy ≥ 95 %** on the existing 1,947-example benchmark
   (`grasp-copilot/evaluation/eval_outputs/paper_benchmark_run001/`). This
   confirms parity with the oracle-trained model on the easy distribution.
2. **WoZ-7B > Oracle-7B on a held-out ambiguous-scenarios set** — defined
   as scenarios where the oracle's hard-coded rules produce demonstrably
   suboptimal behavior (e.g., user motion patterns that don't cleanly map
   to a single candidate; multi-modal yaw oscillation; rapid mode
   switching). Build this set from the held-out 50 + curated additions.
3. **Inter-wizard κ ≥ 0.7** on ask-vs-act decisions.

---

## 11 · Pitfalls to avoid

- ❌ **Do not** let wizards see the full visual workspace, ground-truth
  user intent, or any field beyond the symbolic blob in §2. This destroys
  deployability.
- ❌ **Do not** collect WoZ data only on reach-to-grasp. If Package 2
  (extended skill repertoire) is included, collect across **all** task
  families in a single unified pass so the model learns one consistent
  ask-or-act policy.
- ❌ **Do not** over-engineer the GUI. A clean Tkinter/PyQt window with a
  matplotlib grid panel is sufficient. No 3D, no animations, no scoring,
  no "wizard leaderboard."
- ❌ **Do not** silently change the JSON schema. The wizard output must
  pass `data_generator.oracle.validate_tool_call` unchanged so it slots
  into the existing `prepare_llm_data` pipeline.
- ❌ **Do not** expose decisions across wizards on the held-out 50 — each
  wizard must annotate independently for κ to be meaningful.
- ❌ **Do not** mix wizard and oracle data in the same training run unless
  it is a deliberate ablation (record the mix ratio if so).
