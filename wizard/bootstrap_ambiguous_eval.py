"""Bootstrap the curated *ambiguous-scenario* eval set for the WoZ paper.

This is the load-bearing eval for the central paper claim
(``plans/training_woz_envs.md`` §5.2 / claim #2): on scenes where the
heuristic oracle would fire prematurely, an Oracle→WoZ-LoRA model should
side with the wizard (typically INTERACT) rather than the oracle's
confident-but-wrong execution.

Unlike :mod:`wizard.bootstrap_self_woz`, every record here is a **single
decision point** (not a 5-step episode) and every state is **deliberately
constructed** to be ambiguous along one specific axis. That gives the
paper:

* a category-balanced eval set (oscillation, post-NO drift, multi-candidate,
  unconfirmed-irreversible, amount-unspecified, ...),
* an "oracle would have done X" diagnostic field per record for
  per-category error analysis,
* a reproducible seed-driven generator the reviewers can rerun.

Output layout, one folder per env::

    data/ambiguous_eval_<env>/grasp_gen.jsonl       # PRIME contract rows
    data/ambiguous_eval_<env>/episodes_meta.jsonl   # category, reason, intended_id
    data/ambiguous_eval_<env>/summary.json          # category + tool distribution

Run after generating it through ``llm.prepare_woz_phase2_data`` (or any
custom contract builder) to get an LLM-ready eval JSONL.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from data_generator.oracle import validate_tool_call

from .env.schematic_env import ENVS, EnvConfig, SchematicEnv
from .prompt_factory import build_prompt


WIZARD_ID = "ambiguous_eval_v1"
DEFAULT_N_PER_CATEGORY = 15


# ----------------------------------------------------------------------------
# Category catalog: (category_id, wizard_prompt_type, human-readable reason)
# ----------------------------------------------------------------------------
#
# Every category is engineered so the oracle's heuristic prescribes a
# confident *execution* (APPROACH / ALIGN_YAW / GRAB / STACK / POUR) but a
# careful wizard would issue an INTERACT first. That is the WoZ-vs-oracle
# disagreement the paper has to measure.

CATEGORIES_YCB: List[Tuple[str, str, str]] = [
    ("multi_candidate_in_cell", "candidate_choice",
     "3 candidates clustered in target cell, no recent INTERACT — oracle would APPROACH first candidate; wizard disambiguates."),
    ("post_no_drift", "anything_else",
     "Past dialog rejected target X (NO reply), gripper drifted back near X — oracle would re-APPROACH X; wizard re-opens with anything_else."),
    ("unconfirmed_approach", "confirm",
     "Single candidate, gripper aligned, no recent confirm — oracle would APPROACH; wizard confirms before irreversible motion."),
    ("yaw_mismatch_with_candidates", "candidate_choice",
     "Gripper at object's cell but yaw 180° off, 2+ candidates — oracle would ALIGN_YAW to closest; wizard disambiguates first."),
    ("oscillation_history", "candidate_choice",
     "Gripper history oscillates between 2 cells with one candidate in each — oracle picks most-recent cell; wizard asks."),
]

CATEGORIES_STACKING: List[Tuple[str, str, str]] = [
    ("multi_stack_target", "candidate_choice",
     "Block held, 2+ valid stack targets in adjacent cells — oracle would STACK closest; wizard disambiguates."),
    ("unconfirmed_stack", "confirm_stack",
     "Single clean stack target, no recent confirm — oracle would STACK; wizard confirms (irreversible)."),
    ("post_no_drift_stack", "anything_else",
     "Past dialog rejected a stack target, gripper drifted back — oracle would re-STACK; wizard re-opens."),
    ("intent_gate_stack_premature", "intent_gate_stack",
     "Block held, candidates not pruned, gripper near targets — oracle would STACK; wizard runs intent gate."),
]

CATEGORIES_POURING: List[Tuple[str, str, str]] = [
    ("amount_unspecified", "amount_choice",
     "Pitcher held near cup, no amount in memory — oracle would POUR with default amount; wizard asks amount_choice."),
    ("multi_cup_target", "candidate_choice",
     "Pitcher held, 2+ candidate cups in same area — oracle would POUR closest; wizard disambiguates."),
    ("pitcher_pre_grab_no_confirm", "pitcher_acquisition",
     "Empty gripper near pitcher, no recent INTERACT — oracle would GRAB; wizard confirms acquisition first."),
    ("unconfirmed_pour_amount", "confirm_amount",
     "Pitcher held, amount staged in last action but not confirmed in dialog — oracle would POUR; wizard confirms."),
    ("cup_full_redirect", "cup_full_redirect",
     "Target cup is FULL; an alternate empty cup is nearby — oracle would POUR into full cup (fail); wizard redirects."),
]


_CATEGORY_BY_ENV: Dict[str, List[Tuple[str, str, str]]] = {
    "reach_to_grasp_ycb": CATEGORIES_YCB,
    "cube_stacking": CATEGORIES_STACKING,
    "pouring": CATEGORIES_POURING,
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _objects_for_env(env: str) -> Tuple[int, int]:
    if env == "reach_to_grasp_ycb":
        return 5, 6
    if env == "cube_stacking":
        return 3, 4
    return 3, 3


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _adjacent_cell(cell: str) -> str:
    # 3x3 grid cells are "A1".."C3"; pick a 4-neighbor with simple shift.
    row, col = cell[0], int(cell[1])
    rows = "ABC"
    r_idx = rows.index(row)
    candidates = []
    if r_idx > 0:
        candidates.append(f"{rows[r_idx - 1]}{col}")
    if r_idx < 2:
        candidates.append(f"{rows[r_idx + 1]}{col}")
    if col > 1:
        candidates.append(f"{row}{col - 1}")
    if col < 3:
        candidates.append(f"{row}{col + 1}")
    return candidates[0]


# ----------------------------------------------------------------------------
# Builder
# ----------------------------------------------------------------------------

class AmbiguousEvalBuilder:
    """Constructs ambiguous-state records for one environment."""

    def __init__(self, env_name: str, *, n_per_category: int, seed: int, output_root: Path):
        if env_name not in _CATEGORY_BY_ENV:
            raise ValueError(f"Unsupported env: {env_name}")
        self.env_name = env_name
        self.n_per_category = n_per_category
        self.seed = seed
        self.rng = random.Random(seed)
        n_min, n_max = _objects_for_env(env_name)
        self.env = SchematicEnv(
            EnvConfig(
                env_name=env_name,
                n_objects_min=n_min,
                n_objects_max=n_max,
                candidate_max_dist=2,
                seed=seed,
            )
        )
        self.categories = _CATEGORY_BY_ENV[env_name]
        self.output_dir = output_root / f"ambiguous_eval_{env_name}"
        self.records: List[Dict[str, Any]] = []
        self.meta_rows: List[Dict[str, Any]] = []

    # ---------------------------------------------------------------- driver

    def build_all(self) -> None:
        for category, prompt_type, reason in self.categories:
            builder = getattr(self, f"_build_{category}", None)
            if builder is None:
                raise NotImplementedError(f"No builder for category {category!r}")
            for idx in range(self.n_per_category):
                self.env.reset()
                try:
                    builder(idx=idx, category=category, prompt_type=prompt_type, reason=reason)
                except _SkipScenario:
                    continue

    def write(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(self.output_dir / "grasp_gen.jsonl", self.records)
        _write_jsonl(self.output_dir / "episodes_meta.jsonl", self.meta_rows)
        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(self._summary(), f, indent=2, sort_keys=True)

    def _summary(self) -> Dict[str, Any]:
        cats = Counter(r["ambiguity_category"] for r in self.records)
        prompts = Counter(r["wizard_prompt_type"] for r in self.records)
        oracle_tools = Counter(r["oracle_baseline_call"]["tool"] for r in self.records)
        return {
            "env_name": self.env_name,
            "wizard_id": WIZARD_ID,
            "seed": self.seed,
            "records": len(self.records),
            "n_per_category_target": self.n_per_category,
            "ambiguity_category_distribution": dict(sorted(cats.items())),
            "wizard_prompt_type_distribution": dict(sorted(prompts.items())),
            "oracle_baseline_tool_distribution": dict(sorted(oracle_tools.items())),
            "output_dir": str(self.output_dir),
        }

    # --------------------------------------------------------------- record

    def _record(
        self,
        wizard_call: Dict[str, Any],
        oracle_call: Dict[str, Any],
        *,
        category: str,
        prompt_type: str,
        reason: str,
    ) -> None:
        validate_tool_call(wizard_call, env=self.env_name)
        blob = self.env.public_blob()
        scenario_index = len(self.records)
        record_id = f"ambig_{self.env_name}_{category}_{scenario_index:03d}"
        self.records.append({
            "id": record_id,
            "episode_id": scenario_index,
            "env_name": self.env_name,
            "ambiguity_category": category,
            "wizard_prompt_type": prompt_type,
            "ambiguity_reason": reason,
            "objects": blob["objects"],
            "gripper_hist": blob["gripper_hist"],
            "memory": blob["memory"],
            "user_state": blob["user_state"],
            "target_tool_call": wizard_call,
            "oracle_baseline_call": oracle_call,
        })
        self.meta_rows.append({
            "episode_id": scenario_index,
            "record_id": record_id,
            "env_name": self.env_name,
            "wizard_id": WIZARD_ID,
            "ambiguity_category": category,
            "wizard_prompt_type": prompt_type,
            "ambiguity_reason": reason,
            "intended_obj_id": self.env.intended_obj_id,
            "wizard_tool": wizard_call["tool"],
            "oracle_tool": oracle_call["tool"],
            "generated_at": time.time(),
        })

    # ---------------------------------------------- state-construction helpers

    def _refresh_history(self, *, cell: Optional[str] = None, yaw: Optional[str] = None,
                        z: Optional[str] = None, mix_with: Optional[str] = None) -> None:
        """Repopulate gripper_hist with the (possibly updated) current state.

        If ``mix_with`` is set, alternate between current cell and ``mix_with``
        to fabricate an oscillating history pattern.
        """
        if cell is not None:
            self.env.gripper.cell = cell
        if yaw is not None:
            self.env.gripper.yaw = yaw
        if z is not None:
            self.env.gripper.z = z
        self.env.gripper_hist.clear()
        if mix_with is None:
            for _ in range(self.env.cfg.history_len):
                self.env.gripper_hist.append(self.env._gripper_record())
            return
        # Oscillating history: bounce current ↔ mix_with cells.
        current = self.env.gripper.cell
        cells_seq = []
        for i in range(self.env.cfg.history_len):
            cells_seq.append(current if i % 2 == 0 else mix_with)
        for c in cells_seq:
            self.env.gripper.cell = c
            self.env.gripper_hist.append(self.env._gripper_record())
        self.env.gripper.cell = current

    def _cluster_at(self, cell: str, obj_ids: Sequence[str]) -> None:
        for o in self.env.objects:
            if o.id in obj_ids:
                o.cell = cell

    def _set_candidates(self, ids: Sequence[str]) -> None:
        self.env.memory["candidates"] = list(ids)

    def _clear_dialog(self) -> None:
        self.env.memory["past_dialogs"] = []
        self.env.memory["n_interactions"] = 0
        self.env.memory["last_tool_calls"] = []
        self.env.memory["last_prompt"] = {}

    def _append_no_dialog(self, obj_id: str, *, action_word: str = "approach") -> None:
        label = self._label(obj_id)
        past = list(self.env.memory.get("past_dialogs") or [])
        past.append({
            "prompt": f"Do you want me to {action_word} the {label}?",
            "kind": "QUESTION",
            "choices": ["1) YES", "2) NO"],
            "reply": "2) NO",
        })
        self.env.memory["past_dialogs"] = past[-6:]
        self.env.memory["n_interactions"] = int(self.env.memory.get("n_interactions", 0)) + 1
        last_calls = list(self.env.memory.get("last_tool_calls") or [])
        last_calls.append("INTERACT")
        self.env.memory["last_tool_calls"] = last_calls[-3:]

    def _label(self, obj_id: str) -> str:
        o = next((o for o in self.env.objects if o.id == obj_id), None)
        return o.label if o else obj_id

    def _intended_and_distractors(self, k: int, *, kind: Optional[str] = None) -> Tuple[str, List[str]]:
        target = next(o for o in self.env.objects if o.id == self.env.intended_obj_id)
        pool = [
            o for o in self.env.objects
            if o.id != target.id and not o.is_held
            and (kind is None or o.kind == kind)
        ]
        if len(pool) < k:
            raise _SkipScenario
        distractors = [o.id for o in pool[:k]]
        return target.id, distractors

    def _held(self) -> Optional[str]:
        held = next((o for o in self.env.objects if o.is_held), None)
        return held.id if held else None

    def _opposite_yaw(self, yaw: str) -> str:
        opp = {"N": "S", "NE": "SW", "E": "W", "SE": "NW",
               "S": "N", "SW": "NE", "W": "E", "NW": "SE"}
        return opp.get(yaw, "S")

    # --------------------------------------------------------- YCB builders

    def _build_multi_candidate_in_cell(self, *, idx, category, prompt_type, reason):
        target_id, distractors = self._intended_and_distractors(2)
        target = next(o for o in self.env.objects if o.id == target_id)
        self._cluster_at(target.cell, [target_id, *distractors])
        self._refresh_history(cell=target.cell, yaw=target.yaw, z="MID")
        self._set_candidates([target_id, *distractors])
        self._clear_dialog()
        wizard_call, _ = build_prompt(self.env_name, prompt_type, self.env.public_blob())
        oracle_call = {"tool": "APPROACH", "args": {"obj": target_id}}
        self._record(wizard_call, oracle_call, category=category, prompt_type=prompt_type, reason=reason)

    def _build_post_no_drift(self, *, idx, category, prompt_type, reason):
        target_id, distractors = self._intended_and_distractors(1)
        target = next(o for o in self.env.objects if o.id == target_id)
        self._cluster_at(target.cell, [target_id])
        self._refresh_history(cell=target.cell, yaw=target.yaw, z="MID")
        self._set_candidates([target_id])
        self._clear_dialog()
        # User already said NO to this target earlier.
        self._append_no_dialog(target_id, action_word="approach")
        wizard_call, _ = build_prompt(self.env_name, prompt_type, self.env.public_blob())
        oracle_call = {"tool": "APPROACH", "args": {"obj": target_id}}
        self._record(wizard_call, oracle_call, category=category, prompt_type=prompt_type, reason=reason)

    def _build_unconfirmed_approach(self, *, idx, category, prompt_type, reason):
        target_id, _ = self._intended_and_distractors(0)
        target = next(o for o in self.env.objects if o.id == target_id)
        # Single candidate, gripper near, aligned, no INTERACT yet.
        for o in self.env.objects:
            if o.id != target_id and not o.is_held:
                o.cell = _adjacent_cell(target.cell)
        self._refresh_history(cell=target.cell, yaw=target.yaw, z="MID")
        self._set_candidates([target_id])
        self._clear_dialog()
        wizard_call, _ = build_prompt(self.env_name, prompt_type, self.env.public_blob(),
                                       target_id=target_id, action="APPROACH")
        oracle_call = {"tool": "APPROACH", "args": {"obj": target_id}}
        self._record(wizard_call, oracle_call, category=category, prompt_type=prompt_type, reason=reason)

    def _build_yaw_mismatch_with_candidates(self, *, idx, category, prompt_type, reason):
        target_id, distractors = self._intended_and_distractors(1)
        target = next(o for o in self.env.objects if o.id == target_id)
        self._cluster_at(target.cell, [target_id, *distractors])
        bad_yaw = self._opposite_yaw(target.yaw)
        self._refresh_history(cell=target.cell, yaw=bad_yaw, z="MID")
        self._set_candidates([target_id, *distractors])
        self._clear_dialog()
        wizard_call, _ = build_prompt(self.env_name, prompt_type, self.env.public_blob())
        # Oracle's heuristic: gripper at cell, just align yaw to closest.
        oracle_call = {"tool": "ALIGN_YAW", "args": {"obj": target_id}}
        self._record(wizard_call, oracle_call, category=category, prompt_type=prompt_type, reason=reason)

    def _build_oscillation_history(self, *, idx, category, prompt_type, reason):
        target_id, distractors = self._intended_and_distractors(1)
        target = next(o for o in self.env.objects if o.id == target_id)
        # Put one candidate at target.cell, another at an adjacent cell.
        other_cell = _adjacent_cell(target.cell)
        self._cluster_at(target.cell, [target_id])
        self._cluster_at(other_cell, distractors)
        # Oscillate the gripper history between target.cell and other_cell.
        self.env.gripper.cell = target.cell
        self.env.gripper.yaw = target.yaw
        self.env.gripper.z = "MID"
        self._refresh_history(mix_with=other_cell)
        self._set_candidates([target_id, *distractors])
        self._clear_dialog()
        wizard_call, _ = build_prompt(self.env_name, prompt_type, self.env.public_blob())
        oracle_call = {"tool": "APPROACH", "args": {"obj": target_id}}
        self._record(wizard_call, oracle_call, category=category, prompt_type=prompt_type, reason=reason)

    # ----------------------------------------------------- Stacking builders

    def _build_multi_stack_target(self, *, idx, category, prompt_type, reason):
        held_id = self._held()
        if not held_id:
            raise _SkipScenario
        free = [o for o in self.env.objects if not o.is_held]
        if len(free) < 2:
            raise _SkipScenario
        primary = free[0]
        secondary = free[1]
        # Place both candidate stack targets in adjacent cells.
        primary.cell = self.env.gripper.cell
        secondary.cell = _adjacent_cell(primary.cell)
        self._refresh_history(cell=primary.cell, yaw=primary.yaw, z="MID")
        self._set_candidates([primary.id, secondary.id])
        self._clear_dialog()
        wizard_call, _ = build_prompt(self.env_name, prompt_type, self.env.public_blob())
        oracle_call = {"tool": "STACK", "args": {"obj": primary.id}}
        self._record(wizard_call, oracle_call, category=category, prompt_type=prompt_type, reason=reason)

    def _build_unconfirmed_stack(self, *, idx, category, prompt_type, reason):
        held_id = self._held()
        if not held_id:
            raise _SkipScenario
        target = next((o for o in self.env.objects if not o.is_held), None)
        if target is None:
            raise _SkipScenario
        for o in self.env.objects:
            if o.id != target.id and not o.is_held:
                o.cell = _adjacent_cell(target.cell)
        self._refresh_history(cell=target.cell, yaw=target.yaw, z="MID")
        self._set_candidates([target.id])
        self._clear_dialog()
        wizard_call, _ = build_prompt(self.env_name, prompt_type, self.env.public_blob(), target_id=target.id)
        oracle_call = {"tool": "STACK", "args": {"obj": target.id}}
        self._record(wizard_call, oracle_call, category=category, prompt_type=prompt_type, reason=reason)

    def _build_post_no_drift_stack(self, *, idx, category, prompt_type, reason):
        held_id = self._held()
        if not held_id:
            raise _SkipScenario
        target = next((o for o in self.env.objects if not o.is_held), None)
        if target is None:
            raise _SkipScenario
        self._refresh_history(cell=target.cell, yaw=target.yaw, z="MID")
        self._set_candidates([target.id])
        self._clear_dialog()
        self._append_no_dialog(target.id, action_word="stack on")
        wizard_call, _ = build_prompt(self.env_name, prompt_type, self.env.public_blob())
        oracle_call = {"tool": "STACK", "args": {"obj": target.id}}
        self._record(wizard_call, oracle_call, category=category, prompt_type=prompt_type, reason=reason)

    def _build_intent_gate_stack_premature(self, *, idx, category, prompt_type, reason):
        held_id = self._held()
        if not held_id:
            raise _SkipScenario
        free = [o for o in self.env.objects if not o.is_held]
        if len(free) < 2:
            raise _SkipScenario
        primary, secondary = free[:2]
        primary.cell = self.env.gripper.cell
        secondary.cell = _adjacent_cell(primary.cell)
        self._refresh_history(cell=primary.cell, yaw=primary.yaw, z="MID")
        # Crucial: candidates not pruned, no INTERACT yet.
        self._set_candidates([primary.id, secondary.id])
        self._clear_dialog()
        wizard_call, _ = build_prompt(self.env_name, prompt_type, self.env.public_blob())
        oracle_call = {"tool": "STACK", "args": {"obj": primary.id}}
        self._record(wizard_call, oracle_call, category=category, prompt_type=prompt_type, reason=reason)

    # ------------------------------------------------------ Pouring builders

    def _build_amount_unspecified(self, *, idx, category, prompt_type, reason):
        # Force pitcher held, cup as intended target, no amount in memory.
        pitcher = next((o for o in self.env.objects if o.kind == "pitcher"), None)
        cups = [o for o in self.env.objects if o.kind == "cup"]
        if pitcher is None or not cups:
            raise _SkipScenario
        cup = cups[0]
        cup.fill = "EMPTY"
        pitcher.is_held = True
        pitcher.cell = cup.cell
        self._refresh_history(cell=cup.cell, yaw=cup.yaw, z="MID")
        self._set_candidates([cup.id])
        self._clear_dialog()
        wizard_call, _ = build_prompt(self.env_name, prompt_type, self.env.public_blob(), target_id=cup.id)
        oracle_call = {"tool": "POUR", "args": {"obj": cup.id, "amount": "FULL"}}
        self._record(wizard_call, oracle_call, category=category, prompt_type=prompt_type, reason=reason)

    def _build_multi_cup_target(self, *, idx, category, prompt_type, reason):
        pitcher = next((o for o in self.env.objects if o.kind == "pitcher"), None)
        cups = [o for o in self.env.objects if o.kind == "cup"]
        if pitcher is None or len(cups) < 2:
            raise _SkipScenario
        cup_a, cup_b = cups[:2]
        for c in (cup_a, cup_b):
            c.fill = "EMPTY"
        pitcher.is_held = True
        pitcher.cell = cup_a.cell
        cup_b.cell = _adjacent_cell(cup_a.cell)
        self._refresh_history(cell=cup_a.cell, yaw=cup_a.yaw, z="MID")
        self._set_candidates([cup_a.id, cup_b.id])
        self._clear_dialog()
        wizard_call, _ = build_prompt(self.env_name, prompt_type, self.env.public_blob())
        oracle_call = {"tool": "POUR", "args": {"obj": cup_a.id, "amount": "FULL"}}
        self._record(wizard_call, oracle_call, category=category, prompt_type=prompt_type, reason=reason)

    def _build_pitcher_pre_grab_no_confirm(self, *, idx, category, prompt_type, reason):
        # Ensure empty gripper near a pitcher with no recent INTERACT.
        pitcher = next((o for o in self.env.objects if o.kind == "pitcher"), None)
        if pitcher is None:
            raise _SkipScenario
        for o in self.env.objects:
            o.is_held = False
        self._refresh_history(cell=pitcher.cell, yaw=pitcher.yaw, z="MID")
        self._set_candidates([pitcher.id])
        self._clear_dialog()
        wizard_call, _ = build_prompt(self.env_name, prompt_type, self.env.public_blob(), target_id=pitcher.id)
        oracle_call = {"tool": "GRAB", "args": {"obj": pitcher.id}}
        self._record(wizard_call, oracle_call, category=category, prompt_type=prompt_type, reason=reason)

    def _build_unconfirmed_pour_amount(self, *, idx, category, prompt_type, reason):
        pitcher = next((o for o in self.env.objects if o.kind == "pitcher"), None)
        cups = [o for o in self.env.objects if o.kind == "cup"]
        if pitcher is None or not cups:
            raise _SkipScenario
        cup = cups[0]
        cup.fill = "EMPTY"
        pitcher.is_held = True
        pitcher.cell = cup.cell
        self._refresh_history(cell=cup.cell, yaw=cup.yaw, z="MID")
        self._set_candidates([cup.id])
        self._clear_dialog()
        chosen_amount = self.rng.choice(["SMALL", "HALF", "FULL"])
        # Stage amount in last_action so the oracle thinks it's ready to POUR.
        self.env.memory["last_action"] = {
            "tool": "ALIGN_YAW", "obj": cup.id, "outcome": "success", "amount": chosen_amount,
        }
        wizard_call, _ = build_prompt(self.env_name, prompt_type, self.env.public_blob(),
                                       target_id=cup.id, amount=chosen_amount)
        oracle_call = {"tool": "POUR", "args": {"obj": cup.id, "amount": chosen_amount}}
        self._record(wizard_call, oracle_call, category=category, prompt_type=prompt_type, reason=reason)

    def _build_cup_full_redirect(self, *, idx, category, prompt_type, reason):
        pitcher = next((o for o in self.env.objects if o.kind == "pitcher"), None)
        cups = [o for o in self.env.objects if o.kind == "cup"]
        if pitcher is None or len(cups) < 2:
            raise _SkipScenario
        full_cup, alt_cup = cups[:2]
        full_cup.fill = "FULL"
        alt_cup.fill = "EMPTY"
        pitcher.is_held = True
        pitcher.cell = full_cup.cell
        alt_cup.cell = _adjacent_cell(full_cup.cell)
        self._refresh_history(cell=full_cup.cell, yaw=full_cup.yaw, z="MID")
        self._set_candidates([full_cup.id, alt_cup.id])
        self._clear_dialog()
        wizard_call, _ = build_prompt(self.env_name, prompt_type, self.env.public_blob(),
                                       target_id=full_cup.id, alt_target_id=alt_cup.id)
        oracle_call = {"tool": "POUR", "args": {"obj": full_cup.id, "amount": "FULL"}}
        self._record(wizard_call, oracle_call, category=category, prompt_type=prompt_type, reason=reason)


class _SkipScenario(Exception):
    """Raised by a builder when the current env.reset() can't produce the needed shape."""


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the curated ambiguous-scenario eval set for the WoZ paper "
            "(Table II in plans/training_woz_envs.md §5.2)."
        )
    )
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--n-per-category", type=int, default=DEFAULT_N_PER_CATEGORY,
                        help="How many scenarios to generate per category per env.")
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--env", choices=[*ENVS, "all"], default="all")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    envs = ENVS if args.env == "all" else (args.env,)
    summaries = []
    for idx, env_name in enumerate(envs):
        builder = AmbiguousEvalBuilder(
            env_name,
            n_per_category=int(args.n_per_category),
            seed=int(args.seed) + idx * 1009,
            output_root=args.output_root,
        )
        builder.build_all()
        builder.write()
        summaries.append(builder._summary())

    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
