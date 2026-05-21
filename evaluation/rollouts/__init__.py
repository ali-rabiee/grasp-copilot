"""
Episode-level rollouts for the noise-robustness evaluation.

This sub-package consumes a `Scenario` corpus (`evaluation/scenarios/`) and
runs forward simulations through the lightweight 3x3 grid sim in
`data_generator/episode.py`, composing:

  * `noise.NoiseProfile` / `NoiseInjector` — four perturbation channels
    (direction, selection, dropout, latency) at literature-grounded levels.
  * `scripted_user.ScriptedUser` — a deterministic target-aware policy that
    drives the gripper toward the labeled target and answers PRIME prompts.
  * `rollout_loop.run_rollout` — the per-(scenario, mode, noise, seed) driver
    that produces per-trial metrics for the sweep.

See `plans/noise-from-real-data.md` §5.3 - §5.6 for the design.
"""

from evaluation.rollouts.noise import NoiseInjector, NoiseProfile, STANDARD_CONDITIONS
from evaluation.rollouts.scripted_user import ScriptedUser
from evaluation.rollouts.rollout_loop import RolloutResult, run_rollout

__all__ = [
    "NoiseInjector",
    "NoiseProfile",
    "STANDARD_CONDITIONS",
    "RolloutResult",
    "ScriptedUser",
    "run_rollout",
]
