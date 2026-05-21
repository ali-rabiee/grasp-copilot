# Perception-noise robustness (existing)

State-perturbation sweep that measures how well a model picks the right tool
when the symbolic state it receives is slightly wrong. Single-timestep, one
LLM call per example.

## What the experiment is

For each (model, perturbation type, noise level p) it runs:

```
contract example ──► perturb symbolic state ──► LLM ──► score tool-call accuracy
   (objects,            (cell jitter, candidate            vs ground truth in
    gripper_hist,        flip, label flip, gripper-          example's `output`
    memory, ...)         history jitter, ...)              field)
```

Perturbations live in `evaluation/benchmarks/robustness_benchmark.py:
PERTURBATION_REGISTRY`:

| Perturbation | What it does |
|---|---|
| `user_input` | Jitters cells/yaws in the **gripper_hist** (recent gripper trajectory) — mimics joystick/head-array input noise as observed by the perception system |
| `grid_jitter` | Jitters object **and** gripper cells — mimics camera-side localization noise |
| `candidate_perturb` | Drops/adds object ids in `memory.candidates` — mimics flaky candidate detection |
| `label_noise` | Replaces object labels with random other labels — mimics misclassification |

For each (perturbation, p) the script reports tool accuracy / strict-exact / motion-object accuracy for the model, plus the heuristic oracle and H1 baselines for reference.

## What's in this folder

| File | Contents |
|---|---|
| `sweep.csv` | Per-(perturbation, p, system) accuracy numbers, one row each |
| `sweep_baselines.csv` | Same but restricted to heuristic baselines |
| `robustness_curves.{pdf,png}` | Plotted curves for the paper |
| `logs/` | Slurm / stdout logs from the original run |

## How to re-run

```bash
python -m evaluation.benchmarks.run_robustness_sweep \
    --models "Qwen3B-FT=models/qwen2_5_3b_oracle_woz_lora" \
    --perturbations user_input grid_jitter candidate_perturb label_noise \
    --noise_levels 0.0 0.1 0.2 0.3 0.5 \
    --out_dir evaluation/results/robustness/perception_noise
```

(The default `--out_dir` already points here.)

## Limitations to acknowledge in paper text

- **One decision per example.** No multi-step rollout, no closed-loop user
  recovery. A model that gets one step wrong here doesn't get a chance to
  fix it.
- **No model of the human user.** The perturbation only affects what the LLM
  perceives; it does not affect what a user would emit, see, or do next.
- **Sensor-style failure modes only.** Doesn't address motor / decoder /
  dropout failure modes that the user-input-noise sweep covers.

For the latter, see [`../user_input_noise/`](../user_input_noise/).
