"""Test path setup so `import evaluation.rollouts...` works from the package dir."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_PKG = _HERE.parents[1]          # .../evaluation/rollouts
_ROOT = _PKG.parents[1]          # .../grasp-copilot

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
