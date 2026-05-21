"""
Four noise channels for the scenario-seeded robustness sweep.

Per plan §5.5, each channel models a real-world failure mode of the
low-bandwidth interfaces PRIME is designed for (joystick / head-array /
EMG / BCI):

  * **direction** — with prob `p_dir`, replace the commanded (axis, direction)
    with a random adjacent (axis, direction). Models motor imprecision in
    joystick / head-array control.
  * **selection** — with prob `p_sel`, replace the user's answer to a PRIME
    INTERACT prompt with a random other valid choice. Models BCI/EMG decoder
    accuracy in the 90 / 80 / 70% range (Wolpaw 2002, Hochberg 2012,
    Pandarinath 2017).
  * **dropout** — with prob `p_drop`, drop the user command entirely (returns
    None — the rollout logs a missed-input tick).
  * **latency** — adds per-command uniform 100-500 ms delay to the simulated
    clock when enabled. Models decoder lag.

Latency does not change tool-selection accuracy, only completion time. Treat
it as a clock-side effect (the sweep code adds it to `clock_sec`), not as a
perturbation on the command itself.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── command perturbation adjacency tables ──────────────────────────────────

# Translation axis "neighbors" in a head-array / joystick sense: swapping x↔y
# is the most common motor-confusion error; flipping +/- direction is the
# next-most-common.
_AXIS_NEIGHBORS: Dict[str, Tuple[str, ...]] = {
    "x": ("y",),
    "y": ("x",),
    "z": ("y", "x"),
}


@dataclass
class NoiseProfile:
    """Levels for a single noise condition.

    Example::

        NoiseProfile("dir_low",     p_dir=0.10)
        NoiseProfile("sel_high",    p_sel=0.30)
        NoiseProfile("compound_mid", p_dir=0.10, p_sel=0.20, p_drop=0.05, latency=True)
    """

    name: str
    p_dir: float = 0.0
    p_sel: float = 0.0
    p_drop: float = 0.0
    latency: bool = False

    def __post_init__(self) -> None:
        for f, lo, hi in [("p_dir", 0.0, 1.0), ("p_sel", 0.0, 1.0), ("p_drop", 0.0, 1.0)]:
            v = getattr(self, f)
            if not (lo <= v <= hi):
                raise ValueError(f"{f}={v} out of range [{lo}, {hi}]")


# Default sweep conditions for the paper plot (plan §5.5):
# clean / per-channel low / per-channel high / compound mid.
STANDARD_CONDITIONS: List[NoiseProfile] = [
    NoiseProfile("clean"),
    NoiseProfile("dir_low",      p_dir=0.10),
    NoiseProfile("dir_high",     p_dir=0.20),
    NoiseProfile("sel_low",      p_sel=0.10),
    NoiseProfile("sel_high",     p_sel=0.30),
    NoiseProfile("compound_mid", p_dir=0.10, p_sel=0.20, p_drop=0.05, latency=True),
]


# ── injector ────────────────────────────────────────────────────────────────


@dataclass
class NoiseInjector:
    profile: NoiseProfile
    rng: random.Random
    _stats: Dict[str, int] = field(default_factory=lambda: {
        "direction_perturbed": 0,
        "selection_perturbed": 0,
        "dropouts": 0,
        "latency_calls": 0,
    })

    def direction(self, cmd: Optional[Dict]) -> Optional[Dict]:
        """Maybe perturb a user velocity command.

        Returns a possibly-perturbed copy of `cmd` (or None if `cmd` is None).
        Leaves gripper commands untouched (there's no meaningful "adjacent
        gripper toggle").
        """
        if cmd is None or self.profile.p_dir <= 0.0:
            return cmd
        if cmd.get("mode") == "gripper":
            return cmd
        if self.rng.random() >= self.profile.p_dir:
            return cmd

        out = dict(cmd)
        # Either swap the axis to a neighbor, or flip the direction. 50/50 mix
        # so neither failure mode dominates.
        if self.rng.random() < 0.5 and out.get("axis") in _AXIS_NEIGHBORS:
            out["axis"] = self.rng.choice(_AXIS_NEIGHBORS[out["axis"]])
        else:
            out["direction"] = -int(out.get("direction", 1))
        self._stats["direction_perturbed"] += 1
        return out

    def selection(self, reply_idx: int, n_choices: int) -> int:
        """Maybe replace a discrete reply index with a random different one.

        `reply_idx` is 0-based and assumed valid (`0 <= reply_idx < n_choices`).
        """
        if n_choices <= 1 or self.profile.p_sel <= 0.0:
            return reply_idx
        if self.rng.random() >= self.profile.p_sel:
            return reply_idx
        alts = [i for i in range(n_choices) if i != reply_idx]
        if not alts:
            return reply_idx
        self._stats["selection_perturbed"] += 1
        return self.rng.choice(alts)

    def dropout(self, cmd: Optional[Dict]) -> Optional[Dict]:
        """Maybe drop a command entirely. The rollout treats None as a no-op tick."""
        if cmd is None or self.profile.p_drop <= 0.0:
            return cmd
        if self.rng.random() < self.profile.p_drop:
            self._stats["dropouts"] += 1
            return None
        return cmd

    def latency_sec(self) -> float:
        """Per-command delay in seconds, 100-500 ms uniform when latency is on."""
        if not self.profile.latency:
            return 0.0
        self._stats["latency_calls"] += 1
        return self.rng.uniform(0.10, 0.50)

    def stats(self) -> Dict[str, int]:
        """Return a copy of the running perturbation counters."""
        return dict(self._stats)


__all__ = [
    "NoiseProfile",
    "NoiseInjector",
    "STANDARD_CONDITIONS",
]
