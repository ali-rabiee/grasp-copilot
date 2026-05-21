# Robustness evaluations

Two distinct noise-robustness experiments live here. They answer **different
questions** and are **not interchangeable** — please read this top-level
README before consuming either set of results.

```
robustness/
├── perception_noise/       (single-timestep state perturbations on the LLM input)
│   └── README.md
└── user_input_noise/       (episode-level user-channel noise via scripted rollouts)
    ├── README.md
    ├── scenarios/          ← 160 real-user scenarios (corpus, committed to repo)
    └── sweeps/             ← per-sweep rollout outputs (not committed — recomputable)
```

---

## What's the difference?

The short version:

| | `perception_noise/` (existing) | `user_input_noise/` (new) |
|---|---|---|
| **What gets noised** | The symbolic **state** the LLM perceives | The user's **velocity / selection commands** |
| **Where the noise lives** | Between the data file and the LLM | Between the scripted user and the simulator |
| **Length of an evaluation unit** | 1 timestep | A whole episode (up to ~2000 ticks) |
| **What gets scored** | Tool-call accuracy vs ground truth | Task success, completion time, input count, interactions |
| **Ground truth needed?** | Yes (labeled tool call per example) | No (success defined as "target grasped") |
| **Scenes** | Synthetic Episode samples | Real PRIME_LOGS user-study trials |
| **User model** | None — single decision per example | Priors-calibrated scripted user (matches real-user emit rate) |
| **Failure modes modeled** | Sensor / perception drift | Motor imprecision (joystick / head-array), decoder errors (BCI / EMG), missed inputs, latency |
| **Paper claim it defends** | "Models pick the right tool even when state is noisy" | "PRIME's value grows as user input bandwidth shrinks — the low-bandwidth-user motivation" |

In one sentence each:

- **`perception_noise/`** is the existing single-timestep robustness sweep
  (`evaluation/benchmarks/robustness_benchmark.py`). It perturbs the symbolic
  state the LLM receives and measures whether the model still picks the right
  tool. Use it to argue *"the model's decisions are robust to noisy state
  observations."*

- **`user_input_noise/`** is the new episode-level sweep
  (`evaluation/benchmarks/scenario_noise_sweep.py`). It seeds the lightweight
  simulator with real PRIME_LOGS scenarios, drives the gripper via a scripted
  user calibrated to that trial's real behavioral priors, and injects noise on
  the user → robot command channel. Use it to argue *"PRIME helps users with
  restricted input channels (BCI / EMG / joystick) more than Manual control."*

The two are **complementary**, not redundant. The paper's robustness section
should reference both: one is single-step model robustness, the other is
end-to-end task-completion robustness.

---

## Why these are separate folders

The previous "synthetic" name was ambiguous — both experiments inject
synthetic perturbations, just at different stages of the pipeline. The
current names disambiguate:

- *perception* noise = robot-side, sensor-style perturbations on what the
  LLM sees
- *user input* noise = human-side, motor / decoder perturbations on what the
  user emits

The scenarios under `user_input_noise/scenarios/` are seeded from real user
trials in `PRIME_LOGS/manual/` and `PRIME_LOGS/assistive/`. This is the only
half of either experiment that uses real human data — see
`user_input_noise/README.md` for provenance details.

---

## What's in each subfolder

See the per-folder READMEs:

- [`perception_noise/README.md`](perception_noise/README.md)
- [`user_input_noise/README.md`](user_input_noise/README.md)

---

## When to look here

| You want to … | Go to … |
|---|---|
| Build a robustness curve for a paper figure showing the model surviving sensor noise | `perception_noise/` |
| Build a robustness curve for the low-bandwidth-user motivation | `user_input_noise/sweeps/` |
| Inspect the 160-scenario corpus extracted from PRIME_LOGS | `user_input_noise/scenarios/` |
| Submit a Unity job to run the noise sweep on a fine-tuned model | start from `user_input_noise/README.md` |
