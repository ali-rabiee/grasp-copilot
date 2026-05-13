"""Env → (Episode class, oracle.decide_tool, default candidate distance) mapping.

The data-generation loop in ``generate_dataset.py`` dispatches through this
registry so it doesn't have to know which env it's running.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Type

from .episode import Episode as EpisodeYCB
from .episode_pouring import EpisodePouring
from .episode_stacking import EpisodeStacking
from .oracle import OracleState, oracle_decide_tool
from .oracle_pouring import pouring_decide_tool
from .oracle_stacking import stacking_decide_tool

DecideFn = Callable[[Sequence[Dict], Sequence[Dict], Dict, OracleState, dict], Dict]


@dataclass(frozen=True)
class EnvSpec:
    name: str
    episode_cls: type
    decide_fn: DecideFn
    default_candidate_max_dist: int


ENV_REGISTRY: Dict[str, EnvSpec] = {
    "reach_to_grasp_ycb": EnvSpec(
        name="reach_to_grasp_ycb",
        episode_cls=EpisodeYCB,
        decide_fn=oracle_decide_tool,
        default_candidate_max_dist=1,
    ),
    "cube_stacking": EnvSpec(
        name="cube_stacking",
        episode_cls=EpisodeStacking,
        decide_fn=stacking_decide_tool,
        default_candidate_max_dist=2,
    ),
    "pouring": EnvSpec(
        name="pouring",
        episode_cls=EpisodePouring,
        decide_fn=pouring_decide_tool,
        default_candidate_max_dist=2,
    ),
}


def get_spec(env: str) -> EnvSpec:
    if env not in ENV_REGISTRY:
        raise ValueError(f"Unknown env: {env!r}. Available: {sorted(ENV_REGISTRY)}")
    return ENV_REGISTRY[env]


def env_names() -> list[str]:
    return sorted(ENV_REGISTRY)
