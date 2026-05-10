"""3x3 grid panel rendered with matplotlib, embedded in the Tk window.

Shows: objects (colored dots with yaw arrows), the gripper (large marker
with yaw arrow + z badge), and the gripper history trail (faded markers).

This is the only "visual" the wizard receives. It corresponds 1:1 to the
symbolic blob — no extra information is leaked.
"""

from __future__ import annotations

import math
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # default non-interactive backend; FigureCanvasTkAgg overrides
from matplotlib.figure import Figure

from data_generator import grid as gridlib

# 8-bin yaw → unit vector for arrow rendering.
_YAW_VEC = {
    "N":  (0.0, 1.0),
    "NE": (math.sin(math.pi / 4),  math.cos(math.pi / 4)),
    "E":  (1.0, 0.0),
    "SE": (math.sin(math.pi / 4), -math.cos(math.pi / 4)),
    "S":  (0.0, -1.0),
    "SW": (-math.sin(math.pi / 4), -math.cos(math.pi / 4)),
    "W":  (-1.0, 0.0),
    "NW": (-math.sin(math.pi / 4),  math.cos(math.pi / 4)),
}

_OBJ_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                "#bcbd22", "#17becf"]


def _cell_to_xy(cell: str) -> tuple[float, float]:
    c = gridlib.Cell.from_label(cell)
    # Render with row 'A' at top: matplotlib y up → invert row index.
    return (c.c + 0.5, (2 - c.r) + 0.5)


class GridView:
    """Encapsulates the matplotlib Figure + draw routine."""

    def __init__(self):
        self.fig = Figure(figsize=(5.0, 5.0), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self._draw_empty()

    def _draw_empty(self) -> None:
        ax = self.ax
        ax.clear()
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        ax.set_aspect("equal")
        for i in range(4):
            ax.axhline(i, color="#cccccc", lw=0.8)
            ax.axvline(i, color="#cccccc", lw=0.8)
        for r_idx, row in enumerate("ABC"):
            for col in (1, 2, 3):
                cell = f"{row}{col}"
                x, y = _cell_to_xy(cell)
                ax.text(x - 0.45, y + 0.42, cell, fontsize=9, color="#888888")
        ax.set_xticks([])
        ax.set_yticks([])

    def render(self, blob: Dict, intended_obj_id_for_debug: str | None = None) -> None:
        """Draw the current state. ``intended_obj_id_for_debug`` is *only*
        used by an explicit dev-mode flag (never on by default) so wizards
        cannot accidentally see ground truth.
        """
        self._draw_empty()
        ax = self.ax

        objects: List[Dict] = list(blob.get("objects") or [])
        gripper_hist: List[Dict] = list(blob.get("gripper_hist") or [])

        # Group objects per cell to fan them out.
        per_cell: Dict[str, List[Dict]] = {}
        for o in objects:
            per_cell.setdefault(o["cell"], []).append(o)

        for cell, group in per_cell.items():
            cx, cy = _cell_to_xy(cell)
            n = len(group)
            for i, o in enumerate(group):
                # Fan within the cell so overlapping objects are still legible.
                offset_angle = 2 * math.pi * i / max(n, 1)
                ox = cx + 0.18 * math.cos(offset_angle) if n > 1 else cx
                oy = cy + 0.18 * math.sin(offset_angle) if n > 1 else cy
                color = _OBJ_PALETTE[hash(o["id"]) % len(_OBJ_PALETTE)]
                marker = "s" if o.get("is_held") else "o"
                ax.scatter([ox], [oy], s=160, c=color, marker=marker,
                           edgecolors="black", linewidths=0.6, zorder=3)
                vx, vy = _YAW_VEC.get(o["yaw"], (0.0, 1.0))
                ax.plot([ox, ox + 0.22 * vx], [oy, oy + 0.22 * vy],
                        color="black", lw=1.2, zorder=4)
                ax.text(ox, oy - 0.3, f"{o['id']}\n{o['label']}",
                        fontsize=7, ha="center", va="top", color="#333333", zorder=5)

        # Gripper history trail.
        if gripper_hist:
            for i, g in enumerate(gripper_hist[:-1]):
                gx, gy = _cell_to_xy(g["cell"])
                # Slight jitter so a stack of identical history points is visible.
                jitter = 0.03 * (i - len(gripper_hist) / 2)
                ax.scatter([gx + jitter], [gy + jitter], s=40,
                           c="#444444", alpha=0.15 + 0.10 * i,
                           edgecolors="none", zorder=2)

            cur = gripper_hist[-1]
            gx, gy = _cell_to_xy(cur["cell"])
            ax.scatter([gx], [gy], s=320, c="#000000", marker="X",
                       edgecolors="white", linewidths=1.4, zorder=6)
            vx, vy = _YAW_VEC.get(cur["yaw"], (0.0, 1.0))
            ax.plot([gx, gx + 0.32 * vx], [gy, gy + 0.32 * vy],
                    color="#000000", lw=2.0, zorder=7)
            ax.text(gx + 0.32 * vx, gy + 0.32 * vy + 0.04,
                    f"G  z={cur.get('z','?')}",
                    fontsize=8, ha="center", va="bottom",
                    color="#000000", fontweight="bold", zorder=8)

        ax.set_title("Workspace (top-down · row A is far)", fontsize=10)
