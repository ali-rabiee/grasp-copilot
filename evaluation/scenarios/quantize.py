"""
Continuous → discrete quantizers for gripper pose.

Convention (documented so the paper subsection can reference it):
  * Yaw is in radians, with yaw=0 → +x axis (East), increasing CCW.
    The 8 bins are 45° wedges centered on N/NE/E/SE/S/SW/W/NW.
    A yaw of 0 rad maps to "E"; π/2 rad maps to "N"; etc.
  * z (gripper height) thresholds match data_generator/episode.py's intuition:
    HIGH = home-pose height (~0.40 m), MID ≈ approach, LOW ≈ grasp.

Empirical distribution in PRIME_LOGS:
  * Almost every initial gripper pose has z ≈ 0.41 m (home pose) → HIGH.
  * Yaw clusters around -0.26 rad ≈ -15° → bin "E" under our convention.
"""

from __future__ import annotations

import math
from typing import Tuple

from evaluation.scenarios.schema import YAW_BINS, Z_BINS

# Bin centers in radians, in the same order as YAW_BINS.
# E = 0, NE = π/4, N = π/2, ..., wrapping CCW.
_YAW_BIN_CENTERS = {
    "E": 0.0,
    "NE": math.pi / 4,
    "N": math.pi / 2,
    "NW": 3 * math.pi / 4,
    "W": math.pi,           # equivalent to -π
    "SW": -3 * math.pi / 4,
    "S": -math.pi / 2,
    "SE": -math.pi / 4,
}
assert set(_YAW_BIN_CENTERS) == set(YAW_BINS)


def _wrap_to_pi(angle_rad: float) -> float:
    """Wrap to [-π, π]."""
    a = (angle_rad + math.pi) % (2 * math.pi) - math.pi
    # Python's mod is sign-correct for floats, so this lands in [-π, π].
    return a


def yaw_rad_to_bin(yaw_rad: float) -> str:
    """Quantize a continuous yaw (radians) to one of the 8 canonical bins.

    Bins are 45° wide, centered on each cardinal/intercardinal direction.
    """
    a = _wrap_to_pi(float(yaw_rad))
    best_bin = "E"
    best_dist = math.inf
    for name, center in _YAW_BIN_CENTERS.items():
        # cyclic distance between two angles
        d = abs(_wrap_to_pi(a - center))
        if d < best_dist:
            best_dist = d
            best_bin = name
    return best_bin


# Height thresholds (meters). The home pose in the user-study logs is ~0.41 m.
# We treat anything ≥ 0.35 m as HIGH (home), 0.18-0.35 m as MID (approach),
# and < 0.18 m as LOW (grasp height).
_Z_HIGH_MIN = 0.35
_Z_MID_MIN = 0.18


def z_height_to_bin(height_m: float) -> str:
    """Quantize a continuous gripper height to one of {HIGH, MID, LOW}."""
    h = float(height_m)
    if h >= _Z_HIGH_MIN:
        return "HIGH"
    if h >= _Z_MID_MIN:
        return "MID"
    return "LOW"


def cell_index_to_label(grid_cell: int) -> str:
    """Map flat 0..8 cell index (row-major) to A1..C3 label.

    Mirrors the PRIME_LOGS state_snapshot encoding: cell=0 → A1, cell=1 → A2,
    cell=2 → A3, cell=3 → B1, ..., cell=8 → C3.
    """
    if not (0 <= grid_cell <= 8):
        raise ValueError(f"grid_cell out of range: {grid_cell}")
    row = "ABC"[grid_cell // 3]
    col = (grid_cell % 3) + 1
    return f"{row}{col}"


__all__ = [
    "yaw_rad_to_bin",
    "z_height_to_bin",
    "cell_index_to_label",
]
