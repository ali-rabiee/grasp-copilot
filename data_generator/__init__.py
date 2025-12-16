"""
Synthetic dataset generator package for the grasp-copilot project.

This package intentionally mirrors the original flat-module layout so it can be
easily extended (e.g., to a 5x5 grid) while keeping imports stable.
"""

from .episode import Episode, OBJECT_LABELS  # re-export for convenience
from .oracle import OracleState, oracle_decide_tool, validate_tool_call  # re-export


