# Package 3 · Scenario-Seeded Noise-Robustness Evaluation — Implementation Plan

**Status:** Drafted 2026-05-15. Authoritative until the calibration figure in §6 is filled in.

**Owner:** Ali Rabiee. Consumers: paper Section IV-D (new subsection "Noise-robustness via scenario-seeded simulation").

---

## 0 · TL;DR

- Build a **lightweight episode-level simulator** on top of `data_generator/episode.py` (3×3 grid sim, **not** IsaacSim — IsaacSim cannot deliver 30k rollouts in 1–2 days).
- Mine **scenarios** (initial layout + target + control-mode prior) from `PRIME_LOGS/assistive/` — the only logs that contain symbolic scene state. Re-label the **manual** trials offline so they're usable too.
- Add a **scripted user model** that drives the simulator toward the labeled target and responds to PRIME's discrete prompts. Calibrate it against the real-user metrics from `user-study-prime/`.
- Add a **noise injection module** with four channels (direction, selection, dropout, latency) and run a 6-condition × 2-mode (Manual vs PRIME) × 10-seed sweep across all scenarios.
- Output: 3 figures and 1 summary table for the paper's new subsection.

---

## 1 · Data audit — what's actually in `grasp-copilot/PRIME_LOGS/`

Counted on 2026-05-15:

| Slice | Count | Notes |
|---|---|---|
| All trials under `{manual,assistive}/s1..s8/{easy,hard}/trial_*/` | **160** | 117 manual, 43 assistive. (`old/` dir has earlier pilot trials — exclude.) |
| Trials with **non-empty `gui_events.jsonl`** (motion data) | 117 (all manual usable) | Source for mode-switch / direction-reversal / active-time priors. |
| Trials with **non-empty `tool_calls.jsonl`** (assistive only) | **40** | These contain `state_snapshot.objects[*]` → layouts + initial gripper pose. |
| Trials with **populated `success` / `success_object_label`** | **0** | Every `trial_summary.json` has `"success": false` and empty labels — the GUI never wrote them. **Cannot trust the success field.** |
| Trials with `tool_call_count > 0` | 42 | Closest proxy for "PRIME activated"; matches user-study cleaning Step 1. |

**Implications that change the plan:**

1. The "253 successful trials" number from the old draft is wrong for this dataset slice. The realistic **scenario count is ≤ 117**, with full symbolic layouts present in only the 40 assistive trials. The remaining 77 manual trials need **offline target/layout labeling** before they can be used as scenarios.
2. The "user's true target" cannot be read from logs as-is. It must be:
   - For **assistive trials with tool calls**: the `target_object_id` of the **last successful** APPROACH / ALIGN_YAW call (40 trials).
   - For **manual trials and assistive trials with no tool calls**: hand-labeled by scrubbing `camera.mp4` (use `video_meta.json` for FPS) or — cheaper — inferred from the final gripper cell vs. nearest object cell using the reset positions known from the experiment design. Hand-label is the safer default for ≤117 trials.
3. The "existing simulation infrastructure" reference must mean the **3×3 grid Episode** in `data_generator/episode.py`. IsaacSim throughput is ~10 rollouts/min on a 4090; 30k rollouts would take ~50 GPU-days. The Episode sim runs at ≥1k rollouts/min single-threaded.

---

## 2 · Objective (revised)

Quantify how PRIME's advantage scales with **input-channel bandwidth**, parameterized by four noise injection points calibrated to literature ranges for joystick / head-array / EMG / BCI users. Provides simulated bridging evidence that the architecture's value grows in the low-bandwidth regime that motivates the work — partially addressing the gap left by an able-bodied-only user study, **without claiming to replace clinical validation**.

---

## 3 · Architecture overview

```
              PRIME_LOGS/                    Scenario corpus
              ───────────                    ────────────────
   trial_meta.json + tool_calls.jsonl ─►  scenarios.jsonl
   gui_events.jsonl                     {layout, target, mode_prior}
                  │                             │
                  ▼                             ▼
   ScriptedUser  ◄──  NoiseInjector  ─►  EpisodeSim (3×3 grid)
   (target-aware,    (direction, selection,    │
    mode-aware)      dropout, latency)         │
                  │                             │
                  ▼                             ▼
              ┌─────────────────────────────────────┐
              │  RolloutLoop                         │
              │   while not terminal:                │
              │     u_t = user(state, noise)         │
              │     state = sim.step_user(u_t)       │
              │     if mode == PRIME:                │
              │        a_t = LLM(state, mem)         │
              │        if INTERACT: u' = user.reply  │
              │        else: state = sim.apply(a_t)  │
              └─────────────────────────────────────┘
                                │
                                ▼
                     Per-rollout metrics ─► aggregator ─► plots
```

Every component below already has at least a scaffold in the repo. The plan lists what to add per component.

---

## 4 · Deliverables (concrete files)

All paths relative to `grasp-copilot/`. **Bold** entries are net-new files.

| Purpose | File | Status |
|---|---|---|
| Scenario corpus extractor | **`evaluation/scenarios/extract_from_prime_logs.py`** | NEW |
| Target / layout labeling helper | **`evaluation/scenarios/label_targets.py`** + **`evaluation/scenarios/manual_targets.csv`** | NEW (hand-edited CSV) |
| Scenario JSONL schema + loader | **`evaluation/scenarios/__init__.py`** | NEW |
| Scripted user model | **`evaluation/scripted_user.py`** | NEW |
| Noise injection module | **`evaluation/noise.py`** | NEW |
| Episode rollout driver | **`evaluation/rollout_loop.py`** | NEW |
| Sweep runner (the "experimental matrix") | **`evaluation/scenario_noise_sweep.py`** | NEW |
| Aggregation + plotting | **`evaluation/scenario_noise_plots.py`** | NEW |
| LLM ask-or-act backend (PRIME mode) | reuse `llm/inference.py` | EXISTING |
| Lightweight env sim | reuse `data_generator/episode.py` (extend with `step_user_command`) | EXISTING (needs ~30 LoC patch) |
| Heuristic ask-or-act backend (cheap PRIME stand-in / Oracle ablation) | reuse `data_generator/oracle.py` `oracle_decide_tool` | EXISTING |
| Real-user calibration baseline | reuse `user-study-prime/results/all_trials_clean.csv` | EXISTING — re-run `extract_data.py` against in-repo `PRIME_LOGS/` first to confirm it points at the right root |

Outputs land under `evaluation/eval_outputs/scenario_noise/`:

```
scenario_noise/
  scenarios.jsonl                  # the corpus
  scenarios_summary.json           # counts, target-label provenance
  calibration/
    calibration_metrics.csv        # scripted vs real-user, p=0 only
    calibration_fig.pdf            # methods-section figure
  sweep/
    rollouts.parquet               # per-rollout records (~30k rows)
    by_condition.csv               # aggregated by (mode, noise_type, p)
  figures/
    fig_noise_main.pdf             # headline plot
    fig_noise_interactions.pdf     # closed-loop adaptation evidence
```

---

## 5 · Step-by-step implementation

### 5.1 Scenario extraction — `evaluation/scenarios/extract_from_prime_logs.py`

Input: `grasp-copilot/PRIME_LOGS/` root (default; CLI-overridable).
Output: `scenarios.jsonl` (one scenario per line) + `scenarios_summary.json`.

Schema for each scenario record (this is the contract every downstream component reads):

```jsonc
{
  "scenario_id": "manual_s3_easy_trial_20260218_152033",
  "source": {"mode": "manual", "subject": "s3", "difficulty": "easy",
             "trial_id": "trial_20260218_152033"},
  "objects": [                          // initial layout, normalized to 3×3 + 8-bin yaw
    {"id": "obj_1", "label": "mug",       "cell": "B2", "yaw": "N", "is_held": false},
    {"id": "obj_2", "label": "mustard",   "cell": "A3", "yaw": "E", "is_held": false}
  ],
  "gripper_init": {"cell": "C1", "yaw": "S", "z": "HIGH"},
  "target_obj_id": "obj_1",             // see §5.2 for provenance
  "target_label_source": "tool_call|hand_label|nearest_final_cell",
  "user_priors": {                      // computed from gui_events.jsonl
    "translation_share": 0.71,          // fraction of active commands
    "rotation_share": 0.18,
    "gripper_share": 0.11,
    "mean_active_burst_sec": 0.35,
    "mode_switches_per_sec": 0.12,
    "direction_reversals_per_sec": 0.04
  },
  "difficulty": "easy"
}
```

**Extraction logic:**

1. Walk `PRIME_LOGS/{manual,assistive}/s*/{easy,hard}/trial_*/` (mirror `user-study-prime/extract_data.py:183-198`).
2. **Assistive trials with `tool_call_count ≥ 1`:** parse the first record in `tool_calls.jsonl`; map its `state_snapshot.objects[*]` (which already carries `grid_label`, `label`, `id`, `is_held`) into the schema; quantize continuous `yaw` to 8 bins using `data_generator/yaw.py`; map `state_snapshot.gripper` similarly (z-bin from `height` via thresholds calibrated from `data_generator/episode.py` Z_BINS).
3. **Manual trials and assistive trials with 0 tool calls:** layout cannot be reconstructed from logs. Two options:
   - **(a) Hand-label** layout in `evaluation/scenarios/manual_targets.csv` (cols: `trial_id, target_obj_id, obj_<i>_label, obj_<i>_cell, obj_<i>_yaw, gripper_init_cell, gripper_init_yaw`). ~30 min/trial for ≤117 trials = ~1 person-day. **Default.**
   - **(b) Programmatic fallback:** seed layout from a small set of canonical experimental scenes (the user study used a fixed set of object placements per difficulty). Document which canonical scene each trial maps to in `manual_targets.csv`. Cheaper but loses scenario diversity — only use if (a) blows the budget.
4. Compute `user_priors` for every trial from `gui_events.jsonl` (reuse the parser in `user-study-prime/extract_data.py:21-92` — `_count_gui_event_types`). Add `mean_active_burst_sec` and per-second rates (the existing extractor only emits totals).
5. Write summary: per-source counts, target-label provenance histogram, missing fields.

**Hard validation gate:** every scenario must have ≥2 objects, exactly one `target_obj_id` matching an object in the layout, and a non-empty `user_priors`. Drop trials that fail and report in `scenarios_summary.json`.

Expected output: **~100–117 scenarios** after hand-labeling. The plan's "253" is replaced with this realistic number; the paper text must use the actual count.

### 5.2 Target labeling — `evaluation/scenarios/label_targets.py` + CSV

Run order:

1. Auto-label assistive trials with tool calls: target = `target_object_id` of the **last** `tool_calls.jsonl` row where the tool is APPROACH or ALIGN_YAW. If multiple distinct targets in a trial (user changed mind), pick the most recent. Source = `tool_call`.
2. For everything else, the CSV is the source of truth. The script reads `manual_targets.csv`, validates each row against the trial's recoverable scene state, and merges into `scenarios.jsonl`.
3. CSV checking: a tiny QA pass that opens `camera.mp4` first frame via OpenCV alongside `manual_targets.csv` and prints `trial_id ✓` / `trial_id ✗` so the labeler can visually spot-check before committing.

### 5.3 Episode sim patch — `data_generator/episode.py`

The Episode class already exposes `apply_user_motion()` (target-aware noisy teleop toward `intended_obj_id`) and `apply_tool()`. It needs **one new method** plus a constructor switch:

```python
# new constructor mode: seed from a Scenario record instead of random sampling
Episode.from_scenario(scenario_dict, rng) -> Episode

# new step method: apply an explicit user velocity command, not a toward-target heuristic
Episode.step_user_command(axis: str, direction: int, mode: str) -> None
    # axis ∈ {"x","y","z","yaw","gripper"}, mode ∈ {"translation","rotation","gripper"}
    # Mutates gripper_hist by one cell/yaw/z step in the commanded direction.
```

`step_user_command` is the seam where the **NoiseInjector** plugs in — it receives a clean `(axis, direction, mode)` triple from `ScriptedUser`, optionally perturbs it, and then mutates state. Keep the patch ≤ 30 LoC; do not refactor the existing `apply_user_motion` (training data generation depends on it).

### 5.4 Scripted user — `evaluation/scripted_user.py`

```python
class ScriptedUser:
    def __init__(self, scenario: dict, rng: random.Random,
                 hesitation_rate: float = 0.05,
                 mode_switch_cost_sec: float = 0.6):
        ...
    def next_command(self, sim_state) -> dict:
        """Returns {"axis","direction","mode"} or None if waiting on PRIME prompt."""
    def answer_prompt(self, interact_call: dict) -> int:
        """Returns the index of the choice consistent with target intent."""
    def is_done(self, sim_state) -> bool:
        """Target object is_held AND gripper is at target cell with correct yaw."""
```

**Decision policy (kept intentionally simple — reviewers will scrutinize):**

- Compute the **deficit vector** between gripper pose and target pose.
- Always work on the largest deficit dimension first: cell-distance → yaw bins → gripper close. This matches the calibrated `translation/rotation/gripper` shares in `user_priors` only loosely; calibration in §6 adjusts via `hesitation_rate` and `mode_switch_cost_sec`.
- For the deficit dimension, pick the axis+direction that reduces it by 1 unit. Tie-break deterministically (row before col, etc.).
- With probability `hesitation_rate`, emit a one-step reversal of the previous command (mimics direction reversals observed in `gui_events.jsonl`).
- Mode-switch cost: when the chosen mode differs from the current control mode, advance the simulated clock by `mode_switch_cost_sec` before emitting the command (this is what makes "total time" meaningful).
- `answer_prompt`: for QUESTION/CONFIRM that names objects, pick the choice whose object matches `target_obj_id`. For mode/anything-else prompts, follow the canonical positive answer. If the noise-free choice list contains no target-consistent option (PRIME failed to surface it), pick "None of them" and the rollout records a `target_filtered_out` flag.

**No learning.** Deterministic policy, only randomized through `hesitation_rate`. Saves the headache of motivating a learned user model to reviewers.

### 5.5 Noise injection — `evaluation/noise.py`

Four channels, each a callable `(command_or_response, p, rng) -> perturbed`:

| Channel | Function signature | Behavior |
|---|---|---|
| Direction noise | `direction(cmd, p, rng)` | With prob `p`, replace `(axis, direction)` with a random adjacent axis or sign-flipped direction. Adjacency table mirrors `data_generator/grid.neighbors`. |
| Selection noise | `selection(reply_idx, choices_n, p, rng)` | With prob `p`, replace the answered index with a uniformly random different valid index. |
| Dropout | `dropout(cmd, p, rng)` | With prob `p`, return `None` — the rollout records a "no-op tick" and advances time. |
| Latency | `latency(cmd, rng)` | Add per-command delay drawn `Uniform(100, 500) ms` to the simulated clock. Always-on; no `p` parameter. |

`NoiseProfile` aggregates them:

```python
@dataclass
class NoiseProfile:
    name: str                # "p_dir=0.1", "compound_mid", etc.
    p_dir: float = 0.0
    p_sel: float = 0.0
    p_drop: float = 0.0
    latency: bool = False
```

**Levels to sweep (literature anchors):**

- `p_dir ∈ {0, 0.05, 0.10, 0.20}` — joystick + head-array motor-imprecision range (Argall 2018; Jain & Argall 2019; head-array studies 5–15% miss-direction).
- `p_sel ∈ {0, 0.10, 0.20, 0.30}` — corresponds to 90 / 80 / 70 % BCI/EMG decoder accuracy (Wolpaw 2002; Hochberg 2012; Pandarinath 2017).
- `p_drop ∈ {0, 0.05, 0.10}` — packet/missed-detection range from BCI online sessions.
- Latency on/off.

**Conditions:**

| Name | p_dir | p_sel | p_drop | Latency |
|---|---|---|---|---|
| `clean` | 0.0 | 0.0 | 0.0 | off |
| `dir_low` | 0.10 | 0 | 0 | off |
| `dir_high` | 0.20 | 0 | 0 | off |
| `sel_low` | 0 | 0.10 | 0 | off |
| `sel_high` | 0 | 0.30 | 0 | off |
| `compound_mid` | 0.10 | 0.20 | 0.05 | on |

Six conditions; covers each channel in isolation and one realistic compound. This matches the plan's intent but pins down the per-channel breakpoints.

### 5.6 Rollout driver — `evaluation/rollout_loop.py`

```python
def run_rollout(scenario, mode, noise_profile, llm_backend, *, seed, max_ticks=200):
    sim = Episode.from_scenario(scenario, rng=random.Random(seed))
    user = ScriptedUser(scenario, rng=random.Random(seed ^ 0xA5A5))
    noise = NoiseInjector(noise_profile, rng=random.Random(seed ^ 0x5A5A))
    clock = 0.0
    interactions = 0
    inputs_total = 0
    while not user.is_done(sim) and tick < max_ticks:
        if mode == "PRIME" and llm_backend.should_act(sim, memory):
            call = llm_backend(sim, memory)
            if call["tool"] == "INTERACT":
                reply_idx = noise.selection(user.answer_prompt(call), len(call["args"]["choices"]))
                memory.record(call, reply_idx)
                interactions += 1
                continue
            else:
                sim.apply_tool(call)         # APPROACH / ALIGN_YAW
                clock += est_motion_time(call, sim)
                continue
        cmd = user.next_command(sim)
        cmd = noise.dropout(noise.direction(cmd))
        if cmd is None:
            clock += TICK_DT                  # missed input
            continue
        clock += noise.latency()
        sim.step_user_command(**cmd)
        inputs_total += 1
    return RolloutResult(success=user.is_done(sim), completion_time=clock,
                          total_inputs=inputs_total, interactions=interactions, ...)
```

**LLM backends to support, in priority order:**

1. **`oracle`** — `oracle_decide_tool` from `data_generator/oracle.py`. Fast (~10k decisions/s). Use for the calibration sweep and as the Oracle ablation row.
2. **`hf_ft`** — the fine-tuned Qwen via `llm/inference.py`. The headline PRIME numbers come from this. Run with `max_new_tokens=128`, deterministic temperature=0. Throughput on a single GPU is the bottleneck; budget on the rough cost in §8.
3. **`manual`** — no LLM at all; the scripted user drives to completion using all three control modes. Produces the Manual baseline.

Memory book-keeping mirrors `data_generator/episode.py` and the contract format (`memory.past_dialogs`, `memory.candidates`, `memory.last_action`, `memory.excluded_obj_ids`, `memory.last_tool_calls`). Reuse the helper that already builds these inside `data_generator/episode.py`'s episode generator — refactor into `evaluation/_memory.py` so the rollout loop and the dataset generator share code paths.

### 5.7 Sweep runner — `evaluation/scenario_noise_sweep.py`

```bash
python -m evaluation.scenario_noise_sweep \
    --scenarios evaluation/eval_outputs/scenario_noise/scenarios.jsonl \
    --backend hf_ft \
    --model_path models/qwen2_5_7b_instruct_ft \
    --conditions clean dir_low dir_high sel_low sel_high compound_mid \
    --modes manual prime \
    --n_seeds 10 \
    --out_dir evaluation/eval_outputs/scenario_noise/sweep
```

- Parallelize across scenarios (multiprocessing for `manual` and `oracle` backends; sequential per-GPU batch for `hf_ft`).
- Stream results to a Parquet writer to keep memory flat.
- Every row: `(scenario_id, mode, condition, seed, success, completion_time, total_inputs, interactions, mode_switches, direction_reversals, target_filtered_out, terminated_at_max_ticks)`.

**Matrix size with realistic counts:** 117 scenarios × 6 conditions × 2 modes × 10 seeds = **14,040 rollouts** per backend. (The original "30,000" assumed 253 scenarios.)

### 5.8 Aggregation + plots — `evaluation/scenario_noise_plots.py`

Three figures:

1. **Headline `fig_noise_main.pdf`**: x = noise level (one panel per channel: `p_dir`, `p_sel`, `p_drop`, `compound`); y₁ = % reduction in completion time (PRIME vs Manual), y₂ = success-rate gap (PRIME − Manual). Two y-axes or twin panels.
2. **`fig_noise_interactions.pdf`**: x = noise level, y = mean interactions per trial under PRIME, with the same channel-wise panels. Shows closed-loop adaptation.
3. **`fig_calibration.pdf`** (methods section): bar chart of scripted-user-at-zero-noise vs real-user means for completion time, total inputs, success rate, with ±25% bands.

Also dump `by_condition.csv` with paired bootstrap CIs (per scenario, paired across modes within (scenario, seed)). Use a fixed seed for the bootstrap.

---

## 6 · Calibration (gate before running the full sweep)

This is the make-or-break step the original plan correctly flagged. Concrete protocol:

1. Re-run `user-study-prime/extract_data.py` against the in-repo `grasp-copilot/PRIME_LOGS/` (the script currently hard-codes `/home/ali/Data/PRIME_LOGS`; either pass `--data-root` or patch the constant) and re-run `analyze.py` to produce `results/all_trials_clean.csv` aligned to the in-repo data. Record N per cell.
2. Run the rollout driver with `mode=manual, noise_profile=clean, n_seeds=10` across **only Easy scenarios**.
3. Compute scripted-user means for: `effective_duration_sec`, `gui_event_count` (= total inputs), `success_rate`.
4. Compare with real-user manual-easy means from step 1.
5. **Pass criterion:** all three metrics within ±25 % of real means. **If pass:** proceed to §5.7 sweep with absolute-value reporting.
6. **If fail:** restrict the headline plot to **relative** changes vs the clean condition (i.e. percent change in PRIME metrics across noise levels, not absolute completion times). Document the failure honestly in the methods section figure.

Tunable knobs to use when calibrating (do **not** invent new ones):

- `hesitation_rate ∈ {0.02, 0.05, 0.10}`
- `mode_switch_cost_sec ∈ {0.3, 0.6, 1.0}`
- per-step `TICK_DT ∈ {0.2, 0.3, 0.5}` seconds (controls absolute time scale only)

Pick one (`hesitation_rate, mode_switch_cost_sec, TICK_DT`) triple. Document it in `calibration_metrics.csv`. **Do not adapt these per scenario** — that would be overfitting the simulator.

---

## 7 · Validation criteria (paper-claim chain)

In the order the paper subsection will argue:

1. **Calibration passes.** Scripted user at zero noise reproduces real-user manual-easy metrics within ±25%. If not, fall back to relative reporting and say so.
2. **PRIME advantage grows monotonically with noise level on ≥ 2 of the 4 channels.** Headline plot's slope is the load-bearing claim.
3. **Interaction count adapts.** Mean PRIME interactions per trial increases with noise level on at least the selection-noise and compound channels (direct closed-loop evidence).
4. **The compound condition does not collapse PRIME.** Even at `compound_mid` (the realistic low-bandwidth user), PRIME maintains >X% success-rate advantage. (X to be filled after the run, not before.)
5. **Manual baseline degrades at least as fast as PRIME on every channel.** Sanity check; if Manual is more robust on any channel, that's a real finding worth a sentence — but PRIME losing on `p_drop` because it asks more often is plausible and must be reported.

---

## 8 · Compute & schedule

| Stage | Backend | Cost |
|---|---|---|
| Scenario extraction | CPU | <30 min |
| Manual hand-labeling | human | ~1 person-day |
| Calibration sweep (Easy only, manual backend) | CPU | <30 min |
| Full sweep — `oracle` backend (sanity) | CPU, multiprocessing | ~1 h |
| Full sweep — `manual` baseline | CPU | ~30 min |
| Full sweep — `hf_ft` 7B (headline) | 1× H100 | **~12–18 h** (≈ 7k PRIME rollouts × ~30 LLM calls × ~250 ms each, batched) |
| Plotting + table generation | CPU | <30 min |
| **Total wall-clock** | | **2 days** (1 person-day labeling + 1 GPU-day sweep) |

If the H100 sweep is the bottleneck, run `hf_ft` at the **3B** model first to verify the pipeline end-to-end (~6 h), then promote to 7B.

---

## 9 · Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Manual hand-labeling balloons past 1 person-day | Medium | Fall back to the canonical-scene mapping in §5.1 (b). |
| Calibration fails the ±25 % gate | Medium | Use relative reporting; this was always the documented fallback. Do not tune the user model per scenario. |
| Scripted user solves Hard scenarios trivially fast → ceiling effect on Manual baseline at low noise | Medium | Lock `TICK_DT` and `mode_switch_cost_sec` from Easy calibration; let Hard show its true difficulty. If Manual-Hard at clean is already at the time cap, raise `max_ticks` rather than tune the user. |
| `hf_ft` throughput too low for 7k rollouts × 30 calls | Medium-high | (a) Use vLLM-style batched decoding through the `llm/inference.py` helpers (currently single-example). (b) Down-shift the headline to the 3B model and add the 7B as an appendix. |
| Reviewers reject the "scenario-seeded" framing | Low (if framed correctly) | Stick to the language guard in §10. Provide the calibration figure as evidence the simulator is not a self-fulfilling prophecy. |
| Latency channel is trivial / no effect | Medium | Expected — latency adds time but doesn't change tool-selection accuracy. Report it in the appendix, not the headline, if effect is null. |

---

## 10 · Pitfalls to avoid (carry-forward + new)

- **Do not call this "replay."** It is scenario-seeded simulation. The corpus is used as a realistic distribution of task instances, not as a literal trajectory replay. Misframing invites methodological criticism.
- **Do not omit the calibration step.** A scripted user that doesn't match human behavior at zero noise invalidates the whole experiment. The calibration figure goes in the methods section regardless of pass/fail.
- **Do not test only selection noise.** PRIME is structurally well-suited to that condition; the experiment must also stress continuous-control noise. Hence the four-channel design in §5.5.
- **Do not claim this substitutes for clinical validation.** Frame as bridging evidence, with motor-impaired user studies as future work.
- **Do not over-fit the scripted user.** One global parameter triple from §6, calibrated on Easy and reused everywhere. If Hard looks wrong, that's a real result, not a bug.
- **Do not silently drop scenarios.** Every dropped trial is logged in `scenarios_summary.json` with a reason. Reviewers will ask why N ≠ 160.
- **Do not trust the `success` field in `trial_summary.json`.** It is empty for all 160 trials. All real-user success comparisons must go through the `user-study-prime/results/all_trials_clean.csv` pipeline.

---

## 11 · Key files and pointers

- **In-repo log root**: `grasp-copilot/PRIME_LOGS/{manual,assistive}/s1..s8/{easy,hard}/trial_*/`
- **Real-user analysis (calibration target)**: `user-study-prime/extract_data.py` (re-point `DATA_ROOT`), `user-study-prime/analyze.py`, `user-study-prime/results/all_trials_clean.csv`
- **Episode sim to extend**: `grasp-copilot/data_generator/episode.py`
- **Grid + yaw quantization**: `grasp-copilot/data_generator/grid.py`, `grasp-copilot/data_generator/yaw.py`
- **Oracle (cheap PRIME stand-in)**: `grasp-copilot/data_generator/oracle.py:oracle_decide_tool`
- **LLM inference**: `grasp-copilot/llm/inference.py`
- **Existing single-timestep perturbation patterns to mirror**: `grasp-copilot/evaluation/robustness_benchmark.py`
- **Models**: `grasp-copilot/models/qwen2_5_7b_instruct_ft`, `grasp-copilot/models/qwen2_5_3b_instruct_ft`

---

## 12 · Open questions

1. **Manual-trial labeling cost.** Is a person-day of hand-labeling acceptable? If not, do we accept the canonical-scene fallback (§5.1 b) and the loss of scenario diversity?
2. **Headline model.** 7B (matches the paper's user study) or 3B (matches the deployable target and is 3× cheaper to sweep)? Default 7B; reassess after the calibration sweep finishes.
3. **Should the "success" gate be re-derived from the videos?** A separate hand-labeling pass over `camera.mp4` (~5 min/trial) would give a real success column. Possibly worth it for the calibration step only.
4. **Hard scenarios with multi-target ambiguity.** If `manual_targets.csv` lists multiple plausible targets for a Hard trial, do we duplicate the scenario (one per target) or pick one canonical target? Default: pick the most-frequent target across the experimenter's notes; document the choice in `scenarios_summary.json`.
