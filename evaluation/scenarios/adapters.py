"""
Adapters from Scenario records to other formats used in this repo.

The two main consumers downstream are:

1.  The **rollout loop** (`evaluation/rollout_loop.py`, planned), which seeds
    `data_generator.Episode` with a scenario and steps it forward. It wants a
    plain Python dict shaped like the LLM input contract.

2.  The **offline evaluation scripts** (`evaluation/offline_exec_benchmark.py`,
    `evaluation/robustness_benchmark.py`, `evaluation/run_full_benchmark.py`)
    which all read **contract JSONL** with the shape:

        {"id": ..., "instruction": ..., "input": "<json-string>", "output": "<json-string>"}

    where `input` is a JSON-encoded dict with keys
    `objects / gripper_hist / memory / user_state`. This adapter exports
    scenarios into that exact shape so they can be dropped into any existing
    evaluator without code changes.

Conventions for t=0 seeding (matches `data_generator/episode.py`):
  * gripper history: the scenario records a single initial pose; we replicate
    it 6 times so models trained on the 6-pose contract receive a fixed-length
    history that signals "no motion yet."
  * memory: empty dialog, no past tool calls, no excluded objects, no
    last_prompt. Candidates are computed deterministically from the layout +
    gripper cell via Manhattan distance ≤ `candidate_max_dist` (mirrors
    `Episode.gripper_candidates`).
  * user_state.mode: defaults to "translation" (the modal starting mode in
    PRIME_LOGS); pass `user_mode=` to override.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from evaluation.scenarios.schema import Scenario


# ── shared constants ────────────────────────────────────────────────────

DEFAULT_INSTRUCTION = (
    "Given the robot observation and dialog context, infer the user's intent and "
    "emit exactly one tool call. Output ONLY the tool call JSON with keys tool and args. "
    "If the tool is INTERACT, you must output at most 5 choices total."
)
DEFAULT_CANDIDATE_MAX_DIST = 1   # mirrors data_generator/episode.gripper_candidates
DEFAULT_USER_MODE = "translation"
GRIPPER_HIST_LEN = 6             # contract input expects a 6-pose history


# ── grid-Manhattan (duplicated locally to avoid cross-package import) ───

_ROW_INDEX = {"A": 0, "B": 1, "C": 2}


def _manhattan(a: str, b: str) -> int:
    return abs(_ROW_INDEX[a[0]] - _ROW_INDEX[b[0]]) + abs(int(a[1]) - int(b[1]))


# ── core conversion ─────────────────────────────────────────────────────


def scenario_to_input_dict(
    scenario: Scenario,
    user_mode: str = DEFAULT_USER_MODE,
    candidate_max_dist: int = DEFAULT_CANDIDATE_MAX_DIST,
) -> Dict[str, Any]:
    """Convert a Scenario into the dict the LLM input contract expects.

    Returned shape (matches `llm/data.py:155-170` and the existing
    contract JSONL files under `data/`):

        {
          "objects": [{"id", "label", "cell", "yaw", "is_held"}, ...],
          "gripper_hist": [{"cell", "yaw", "z"}] * 6,
          "memory": {
              "n_interactions": 0,
              "past_dialogs": [],
              "candidates": [...],
              "last_tool_calls": [],
              "excluded_obj_ids": [],
              "last_action": {},
          },
          "user_state": {"mode": "translation"}
        }
    """
    # Drop the schema's bookkeeping fields (raw_label) that aren't part of the
    # contract — keep only what the LLM was trained to see.
    objects = [
        {
            "id": o.id,
            "label": o.label,
            "cell": o.cell,
            "yaw": o.yaw,
            "is_held": o.is_held,
        }
        for o in scenario.objects
    ]

    pose = {
        "cell": scenario.gripper_init.cell,
        "yaw": scenario.gripper_init.yaw,
        "z": scenario.gripper_init.z,
    }
    gripper_hist = [dict(pose) for _ in range(GRIPPER_HIST_LEN)]

    # Candidates at t=0: objects within candidate_max_dist of the gripper,
    # excluding held objects.
    candidates = [
        o["id"]
        for o in objects
        if not o["is_held"]
        and _manhattan(scenario.gripper_init.cell, o["cell"]) <= candidate_max_dist
    ]

    memory = {
        "n_interactions": 0,
        "past_dialogs": [],
        "candidates": candidates,
        "last_tool_calls": [],
        "excluded_obj_ids": [],
        "last_action": {},
    }

    return {
        "objects": objects,
        "gripper_hist": gripper_hist,
        "memory": memory,
        "user_state": {"mode": user_mode},
    }


def scenario_to_contract_row(
    scenario: Scenario,
    instruction: str = DEFAULT_INSTRUCTION,
    user_mode: str = DEFAULT_USER_MODE,
    candidate_max_dist: int = DEFAULT_CANDIDATE_MAX_DIST,
    output_tool_call: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Convert a Scenario into a contract JSONL row.

    `output_tool_call` is optional; pass it to fill the `output` field with a
    ground-truth (e.g. the oracle's decision at t=0). When None, the output
    field is an empty JSON object — usable for inference but not for
    accuracy-based scoring.
    """
    input_dict = scenario_to_input_dict(scenario, user_mode=user_mode, candidate_max_dist=candidate_max_dist)
    return {
        "id": scenario.scenario_id,
        "instruction": instruction,
        "input": json.dumps(input_dict, ensure_ascii=False),
        "output": json.dumps(output_tool_call or {}, ensure_ascii=False),
    }


# ── bulk export ─────────────────────────────────────────────────────────


def write_scenarios_as_contract_jsonl(
    scenarios: Iterable[Scenario],
    out_path: str | Path,
    *,
    instruction: str = DEFAULT_INSTRUCTION,
    user_mode: str = DEFAULT_USER_MODE,
    candidate_max_dist: int = DEFAULT_CANDIDATE_MAX_DIST,
    skip_unlabeled: bool = False,
    output_tool_call_fn=None,
) -> int:
    """Export scenarios as a contract JSONL file.

    Args:
        scenarios: iterable of Scenario records.
        out_path: destination JSONL path.
        instruction: instruction string written into every row.
        user_mode: initial user_state.mode value.
        candidate_max_dist: Manhattan radius for the t=0 candidate set.
        skip_unlabeled: drop scenarios whose target_obj_id is None (i.e. not
            yet hand-labeled). Useful when the downstream evaluator needs a
            ground-truth target.
        output_tool_call_fn: optional callable
            `(scenario, input_dict) -> dict | None`. Lets the caller compute a
            per-scenario `output` (e.g. by running the heuristic oracle).
            When None, every row gets an empty `output`.

    Returns the number of rows written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for s in scenarios:
            if skip_unlabeled and s.target_obj_id is None:
                continue
            input_dict = scenario_to_input_dict(
                s, user_mode=user_mode, candidate_max_dist=candidate_max_dist
            )
            output_call = output_tool_call_fn(s, input_dict) if output_tool_call_fn else None
            row = {
                "id": s.scenario_id,
                "instruction": instruction,
                "input": json.dumps(input_dict, ensure_ascii=False),
                "output": json.dumps(output_call or {}, ensure_ascii=False),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


# ── oracle output helper (optional) ─────────────────────────────────────


def oracle_output_for(scenario: Scenario, input_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Compute the heuristic oracle's t=0 decision for a scenario.

    Returns a tool-call dict suitable for the contract `output` field. Falls
    back to None if the oracle errors (e.g. unsupported env) — leave that to
    the caller to handle.

    Requires the user to have already labeled `scenario.target_obj_id`; the
    oracle is target-aware and cannot guess.
    """
    if scenario.target_obj_id is None:
        return None
    try:
        from data_generator.oracle import OracleState, oracle_decide_tool
    except Exception:
        return None

    state = OracleState(intended_obj_id=scenario.target_obj_id)
    try:
        return oracle_decide_tool(
            input_dict["objects"],
            input_dict["gripper_hist"],
            input_dict["memory"],
            state,
            user_state=input_dict["user_state"],
        )
    except Exception:
        return None


__all__ = [
    "DEFAULT_INSTRUCTION",
    "DEFAULT_CANDIDATE_MAX_DIST",
    "DEFAULT_USER_MODE",
    "GRIPPER_HIST_LEN",
    "scenario_to_input_dict",
    "scenario_to_contract_row",
    "write_scenarios_as_contract_jsonl",
    "oracle_output_for",
]
