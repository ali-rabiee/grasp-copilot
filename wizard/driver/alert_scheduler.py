"""Decide when to interrupt the simulated rollout for a wizard decision.

Stochastic-plus-event schedule (see README §4.3):

  - Episode start (always).
  - After every execution skill returns (always).
  - When the candidate-set size changes vs. the previous tick (always).
  - Otherwise, with probability ``p_alert`` per tick.
"""

from __future__ import annotations

import enum
import random
from dataclasses import dataclass
from typing import List, Optional


class AlertReason(enum.Enum):
    EPISODE_START = "episode_start"
    POST_EXECUTION = "post_execution"
    CANDIDATE_CHANGE = "candidate_change"
    STOCHASTIC = "stochastic"


@dataclass
class AlertScheduler:
    p_alert: float = 0.15
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._prev_candidates: Optional[List[str]] = None
        self._just_executed = False
        self._is_first_tick = True

    def reset_episode(self) -> None:
        self._prev_candidates = None
        self._just_executed = False
        self._is_first_tick = True

    def mark_post_execution(self) -> None:
        self._just_executed = True

    def should_alert(self, candidates: List[str]) -> Optional[AlertReason]:
        if self._is_first_tick:
            self._is_first_tick = False
            self._prev_candidates = list(candidates)
            return AlertReason.EPISODE_START
        if self._just_executed:
            self._just_executed = False
            self._prev_candidates = list(candidates)
            return AlertReason.POST_EXECUTION
        if self._prev_candidates is not None and set(candidates) != set(self._prev_candidates):
            self._prev_candidates = list(candidates)
            return AlertReason.CANDIDATE_CHANGE
        self._prev_candidates = list(candidates)
        if self._rng.random() < self.p_alert:
            return AlertReason.STOCHASTIC
        return None
