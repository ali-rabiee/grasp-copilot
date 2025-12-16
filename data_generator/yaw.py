from __future__ import annotations

from typing import List, Tuple


YAW_BINS: Tuple[str, ...] = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
_IDX = {b: i for i, b in enumerate(YAW_BINS)}


def normalize(bin_name: str) -> str:
    b = bin_name.upper()
    if b not in _IDX:
        raise ValueError(f"Invalid yaw_bin: {bin_name!r}")
    return b


def cyclic_distance_steps(a: str, b: str) -> int:
    ia, ib = _IDX[normalize(a)], _IDX[normalize(b)]
    d = abs(ia - ib) % len(YAW_BINS)
    return min(d, len(YAW_BINS) - d)


def move_toward(current: str, target: str, steps: int = 1) -> str:
    """
    Moves 1 yaw-bin step per step toward target on a cyclic 8-bin ring.
    Deterministic tie-breaker: move clockwise when equidistant.
    """
    if steps < 0:
        raise ValueError("steps must be >= 0")
    cur, tgt = _IDX[normalize(current)], _IDX[normalize(target)]
    n = len(YAW_BINS)
    for _ in range(steps):
        if cur == tgt:
            break
        cw = (tgt - cur) % n
        ccw = (cur - tgt) % n
        if cw <= ccw:
            cur = (cur + 1) % n
        else:
            cur = (cur - 1) % n
    return YAW_BINS[cur]


def neighbors(bin_name: str) -> List[str]:
    i = _IDX[normalize(bin_name)]
    n = len(YAW_BINS)
    return [YAW_BINS[(i - 1) % n], YAW_BINS[(i + 1) % n]]


