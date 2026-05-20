"""
Scenario corpus extracted from PRIME_LOGS user-study trials.

A *scenario* is a self-contained task instance — initial object layout, initial
gripper pose, the user's target object, and behavioral priors — that downstream
noise-robustness rollouts can seed the lightweight Episode simulator with.

This is **not** a trajectory replay: we keep only what defines the task, not the
moment-to-moment human teleop signal. See `plans/noise-from-real-data.md` §5.1.
"""

from evaluation.scenarios.schema import (
    GripperInit,
    ObjectInit,
    Scenario,
    UserPriors,
    load_scenarios,
    write_scenarios,
)

__all__ = [
    "GripperInit",
    "ObjectInit",
    "Scenario",
    "UserPriors",
    "load_scenarios",
    "write_scenarios",
]
