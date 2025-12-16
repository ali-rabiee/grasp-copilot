from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


ROWS: Tuple[str, ...] = ("A", "B", "C")
COLS: Tuple[int, ...] = (1, 2, 3)
CELLS: Tuple[str, ...] = tuple(f"{r}{c}" for r in ROWS for c in COLS)


@dataclass(frozen=True, slots=True)
class Cell:
    r: int  # 0..2
    c: int  # 0..2

    def to_label(self) -> str:
        return f"{ROWS[self.r]}{self.c + 1}"

    @staticmethod
    def from_label(label: str) -> "Cell":
        if len(label) != 2:
            raise ValueError(f"Invalid cell label: {label!r}")
        row, col = label[0].upper(), label[1]
        if row not in ROWS or col not in {"1", "2", "3"}:
            raise ValueError(f"Invalid cell label: {label!r}")
        return Cell(ROWS.index(row), int(col) - 1)


def manhattan(a: str, b: str) -> int:
    ca, cb = Cell.from_label(a), Cell.from_label(b)
    return abs(ca.r - cb.r) + abs(ca.c - cb.c)


def same_row_or_col(a: str, b: str) -> bool:
    ca, cb = Cell.from_label(a), Cell.from_label(b)
    return ca.r == cb.r or ca.c == cb.c


def neighbors(cell: str) -> List[str]:
    c = Cell.from_label(cell)
    out: List[str] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = c.r + dr, c.c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            out.append(Cell(nr, nc).to_label())
    return out


def step_toward(current: str, target: str) -> str:
    """
    Returns a 4-connected Manhattan step that reduces distance to target.
    Deterministic tie-breaking: prefer row moves, then col moves.
    """
    if current == target:
        return current
    cur, tgt = Cell.from_label(current), Cell.from_label(target)
    dr, dc = tgt.r - cur.r, tgt.c - cur.c
    nr, nc = cur.r, cur.c
    if dr != 0:
        nr += 1 if dr > 0 else -1
    else:
        nc += 1 if dc > 0 else -1
    return Cell(nr, nc).to_label()


def nearest_cells_by_distance(
    origin: str, candidates: Iterable[str]
) -> List[Tuple[str, int]]:
    items = [(cell, manhattan(origin, cell)) for cell in candidates]
    items.sort(key=lambda x: (x[1], x[0]))
    return items

