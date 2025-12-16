"""
Test-time path setup.

These tests live under `data_generator/tests/`. Depending on where pytest is invoked
(repo root vs `data_generator/`), the project root may or may not be on `sys.path`.

We make imports deterministic by ensuring the parent directory of `data_generator/`
is on `sys.path`, so `import data_generator` always works.
"""

from __future__ import annotations

import sys
from pathlib import Path


_HERE = Path(__file__).resolve()
_PKG_DIR = _HERE.parents[1]  # .../data_generator
_ROOT = _PKG_DIR.parent      # .../grasp-copilot

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


