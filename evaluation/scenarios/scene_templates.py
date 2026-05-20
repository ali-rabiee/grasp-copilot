"""
Build canonical (subject, difficulty) scene templates from assistive trials.

Manual trials in PRIME_LOGS contain no scene state at all (empty
tool_calls.jsonl, empty llm_events.jsonl). However the experimental protocol
*reset object positions to the same canonical layout* between trials within
the same (subject, difficulty) cell, modulo minor perception jitter. This
module exploits that structure: it scans all *assistive* trials for a given
(subject, difficulty), votes on the most frequent object layout, and produces
a single template that manual trials of the same (subject, difficulty) can
inherit.

The voting is per-(label, cell) plurality: an object exists in the template
if and only if it was observed in cell `c` with canonical label `l` in at
least `min_support` of the assistive trials for that cell.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from evaluation.scenarios.label_normalize import normalize
from evaluation.scenarios.log_parser import RawObject, TrialData, dedup_objects
from evaluation.scenarios.quantize import yaw_rad_to_bin, z_height_to_bin


# Default plurality threshold: keep an (object, cell) pair if it shows up in
# at least 1 assistive trial. Higher values reject noisy detections at the
# cost of throwing out rare-but-correct objects.
DEFAULT_MIN_SUPPORT = 1


@dataclass
class GripperTemplate:
    cell: str
    yaw: str
    z: str


@dataclass
class SceneTemplate:
    subject: str
    difficulty: str
    objects: List[RawObject]  # canonicalized + deduped
    gripper: GripperTemplate
    source_trial_count: int = 0
    notes: str = ""


def _vote_objects(
    assistive_trials: List[TrialData],
    min_support: int,
) -> List[RawObject]:
    """Plurality vote on (canonical_label, cell) pairs across all trials' first snapshots."""
    counter: Counter = Counter()
    # Map each (label, cell) to the most recent representative raw object so we
    # can keep yaw / is_held metadata for the winner.
    rep: Dict[Tuple[str, str], RawObject] = {}

    for td in assistive_trials:
        if not td.tool_calls:
            continue
        first_snap = td.tool_calls[0].state_snapshot
        deduped = dedup_objects(
            [
                RawObject(
                    id=o.id,
                    label=normalize(o.label),
                    grid_cell=o.grid_cell,
                    grid_label=o.grid_label,
                    is_held=o.is_held,
                    yaw_rad=o.yaw_rad,
                )
                for o in first_snap.objects
            ]
        )
        for o in deduped:
            key = (o.label, o.grid_label)
            counter[key] += 1
            rep.setdefault(key, o)

    winners: List[RawObject] = []
    for key, cnt in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
        if cnt < min_support:
            continue
        winners.append(rep[key])

    # Reassign object ids contiguously starting at obj_1, in (cell, label) order
    # so the template is deterministic across builds.
    winners.sort(key=lambda o: (o.grid_label, o.label))
    for i, o in enumerate(winners, start=1):
        o.id = f"obj_{i}"
        o.is_held = False  # templates always start with nothing held
    return winners


def _vote_gripper(assistive_trials: List[TrialData]) -> Optional[GripperTemplate]:
    """Plurality on initial gripper (cell, yaw_bin, z_bin)."""
    counter: Counter = Counter()
    for td in assistive_trials:
        if not td.tool_calls:
            continue
        g = td.tool_calls[0].state_snapshot.gripper
        key = (g.grid_label, yaw_rad_to_bin(g.yaw_rad), z_height_to_bin(g.height_m))
        counter[key] += 1
    if not counter:
        return None
    (cell, yaw, z), _ = counter.most_common(1)[0]
    return GripperTemplate(cell=cell, yaw=yaw, z=z)


def build_templates(
    trials_by_key: Dict[Tuple[str, str], List[TrialData]],
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> Dict[Tuple[str, str], SceneTemplate]:
    """Build {(subject, difficulty) -> SceneTemplate} from assistive trials.

    Only assistive trials contribute. The caller is responsible for filtering
    to assistive trials before passing them in (we don't assume the dict has
    been pre-filtered).
    """
    out: Dict[Tuple[str, str], SceneTemplate] = {}
    for (subject, difficulty), trials in trials_by_key.items():
        assistive = [t for t in trials if t.meta.get("mode") == "assistive" and t.tool_calls]
        if not assistive:
            continue
        objs = _vote_objects(assistive, min_support=min_support)
        gripper = _vote_gripper(assistive)
        if not objs or gripper is None:
            continue
        out[(subject, difficulty)] = SceneTemplate(
            subject=subject,
            difficulty=difficulty,
            objects=objs,
            gripper=gripper,
            source_trial_count=len(assistive),
        )
    return out


def build_difficulty_fallbacks(
    trials_by_key: Dict[Tuple[str, str], List[TrialData]],
    majority_fraction: float = 0.5,
) -> Dict[str, SceneTemplate]:
    """Build {difficulty -> SceneTemplate} by pooling assistive trials across subjects.

    Used for (subject, difficulty) cells where no per-subject assistive data
    exists (only 4/8 subjects have any assistive trials in our dataset).
    The fallback layout reflects "what the protocol set up for this
    difficulty", not any one subject's recorded session.

    To avoid pulling in subject-specific layout *variants* (e.g. some hard
    layouts place gelatin_box@B1 while others place it @B3), we require an
    (object, cell) pair to appear in at least `majority_fraction` of the
    pooled trials. The default 0.5 keeps only majority-supported placements.
    """
    out: Dict[str, SceneTemplate] = {}
    pooled: Dict[str, List[TrialData]] = {"easy": [], "hard": []}
    for (_subject, difficulty), trials in trials_by_key.items():
        for t in trials:
            if t.meta.get("mode") == "assistive" and t.tool_calls:
                pooled.setdefault(difficulty, []).append(t)

    for difficulty, assistive in pooled.items():
        if not assistive:
            continue
        threshold = max(1, int(len(assistive) * majority_fraction))
        objs = _vote_objects(assistive, min_support=threshold)
        gripper = _vote_gripper(assistive)
        if not objs or gripper is None:
            continue
        subjects = sorted({t.meta.get("subject_id", "?") for t in assistive})
        out[difficulty] = SceneTemplate(
            subject="(pooled)",
            difficulty=difficulty,
            objects=objs,
            gripper=gripper,
            source_trial_count=len(assistive),
            notes=f"pooled across {','.join(subjects)} (min support {threshold}/{len(assistive)})",
        )
    return out


__all__ = [
    "DEFAULT_MIN_SUPPORT",
    "GripperTemplate",
    "SceneTemplate",
    "build_templates",
    "build_difficulty_fallbacks",
]
