"""
Extract a scenario corpus from PRIME_LOGS.

Per the noise-from-real-data plan §5.1, every trial under
`PRIME_LOGS/{manual,assistive}/s*/{easy,hard}/trial_*/` is turned into a
Scenario record. Provenance is tracked carefully so the paper text can be
honest about what was auto-derived vs. what came from hand-labels.

Provenance map for the three cases that arise in practice:

| Trial type                              | layout_source          | target_label_source   |
|-----------------------------------------|------------------------|-----------------------|
| Assistive w/ tool_calls (40 trials)     | state_snapshot         | tool_call             |
| Assistive w/o tool_calls (rare)         | borrowed_template      | unlabeled             |
| Manual (no scene state at all, 117)     | borrowed_template      | unlabeled             |

When `--hand_labels` is passed, `target_label_source` and `layout_source` for
rows present in the CSV are upgraded to `hand_label`.

CLI:
    python -m evaluation.scenarios.extract_from_prime_logs \\
        --logs_root PRIME_LOGS \\
        --out_dir evaluation/results/robustness/user_input_noise/scenarios \\
        [--hand_labels evaluation/scenarios/manual_targets.csv]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from evaluation import _bootstrap  # noqa: F401
except Exception:
    import _bootstrap  # type: ignore  # noqa: F401

from evaluation.scenarios.label_normalize import normalize
from evaluation.scenarios.log_parser import (
    RawObject,
    TrialData,
    dedup_objects,
    discover_trials,
    load_trial,
)
from evaluation.scenarios.quantize import yaw_rad_to_bin, z_height_to_bin
from evaluation.scenarios.scene_templates import (
    SceneTemplate,
    build_difficulty_fallbacks,
    build_templates,
)
from evaluation.scenarios.schema import (
    LAYOUT_SOURCE_BORROWED_TEMPLATE,
    LAYOUT_SOURCE_STATE_SNAPSHOT,
    TARGET_SOURCE_TOOL_CALL,
    TARGET_SOURCE_UNLABELED,
    GripperInit,
    ObjectInit,
    Scenario,
    TrialSource,
    UserPriors,
    write_scenarios,
)
from evaluation.scenarios.user_priors import compute_user_priors


# ── per-trial assembly ──────────────────────────────────────────────────


def _raw_to_object_init(r: RawObject) -> ObjectInit:
    return ObjectInit(
        id=r.id,
        label=normalize(r.label),
        raw_label=r.label,
        cell=r.grid_label,
        yaw=yaw_rad_to_bin(r.yaw_rad),
        is_held=r.is_held,
    )


def _infer_target_from_tool_calls(td: TrialData) -> Optional[str]:
    """Target = the most recent APPROACH/ALIGN_YAW target_object_id, if any."""
    for rec in reversed(td.tool_calls):
        if rec.tool_name in ("APPROACH", "ALIGN_YAW") and rec.target_object_id:
            return rec.target_object_id
    return None


def _remap_target_after_dedup(
    target_id: Optional[str],
    pre_dedup_objs: List[RawObject],
    deduped: List[RawObject],
) -> Optional[str]:
    """After dedup reassigns object ids, find the new id for the original target."""
    if target_id is None:
        return None
    # Locate the original target's (label, cell), then find that key in deduped.
    src = next((o for o in pre_dedup_objs if o.id == target_id), None)
    if src is None:
        return None
    key = (normalize(src.label), src.grid_label)
    for o in deduped:
        if (o.label, o.grid_label) == key:
            return o.id
    return None


def _scenario_from_assistive_trial(
    mode: str,
    subject: str,
    difficulty: str,
    td: TrialData,
    logs_root: Path,
) -> Optional[Scenario]:
    if not td.tool_calls:
        return None
    first_snap = td.tool_calls[0].state_snapshot

    # 1) normalize labels then dedup duplicate detections
    pre = [
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
    deduped = dedup_objects(pre)

    target_orig = _infer_target_from_tool_calls(td)
    target_id = _remap_target_after_dedup(target_orig, pre, deduped)

    objects = [_raw_to_object_init(o) for o in deduped]
    gripper = GripperInit(
        cell=first_snap.gripper.grid_label,
        yaw=yaw_rad_to_bin(first_snap.gripper.yaw_rad),
        z=z_height_to_bin(first_snap.gripper.height_m),
    )

    priors = compute_user_priors(td.gui_events)

    trial_id = td.meta.get("trial_id") or td.trial_dir.name
    scenario_id = f"{mode}_{subject}_{difficulty}_{trial_id}"
    src = TrialSource(
        mode=mode,
        subject=subject,
        difficulty=difficulty,
        trial_id=trial_id,
        trial_dir=str(td.trial_dir.relative_to(logs_root.parent)) if td.trial_dir.is_relative_to(logs_root.parent) else str(td.trial_dir),
    )

    return Scenario(
        scenario_id=scenario_id,
        source=src,
        objects=objects,
        gripper_init=gripper,
        target_obj_id=target_id,
        target_label_source=TARGET_SOURCE_TOOL_CALL if target_id else TARGET_SOURCE_UNLABELED,
        layout_source=LAYOUT_SOURCE_STATE_SNAPSHOT,
        user_priors=priors,
        difficulty=difficulty,
        plausible_target_ids=[o.id for o in objects if not o.is_held],
        notes="",
    )


def _scenario_from_template(
    mode: str,
    subject: str,
    difficulty: str,
    td: TrialData,
    template: SceneTemplate,
    logs_root: Path,
) -> Scenario:
    """Build a scenario by inheriting the canonical (subj, diff) scene template."""
    objects = [_raw_to_object_init(o) for o in template.objects]
    gripper = GripperInit(
        cell=template.gripper.cell,
        yaw=template.gripper.yaw,
        z=template.gripper.z,
    )
    priors = compute_user_priors(td.gui_events)

    trial_id = td.meta.get("trial_id") or td.trial_dir.name
    scenario_id = f"{mode}_{subject}_{difficulty}_{trial_id}"
    src = TrialSource(
        mode=mode,
        subject=subject,
        difficulty=difficulty,
        trial_id=trial_id,
        trial_dir=str(td.trial_dir.relative_to(logs_root.parent)) if td.trial_dir.is_relative_to(logs_root.parent) else str(td.trial_dir),
    )

    return Scenario(
        scenario_id=scenario_id,
        source=src,
        objects=objects,
        gripper_init=gripper,
        target_obj_id=None,
        target_label_source=TARGET_SOURCE_UNLABELED,
        layout_source=LAYOUT_SOURCE_BORROWED_TEMPLATE,
        user_priors=priors,
        difficulty=difficulty,
        plausible_target_ids=[o.id for o in objects if not o.is_held],
        notes=f"layout inherited from {template.source_trial_count} assistive {subject}/{difficulty} trials",
    )


# ── hand-label CSV merge ────────────────────────────────────────────────


def _apply_hand_labels(scenarios: List[Scenario], csv_path: Path) -> int:
    """In-place overlay of `manual_targets.csv` onto scenarios.

    The CSV's contract is documented in `manual_targets.csv` itself. Only the
    target_obj_id is read here; whole-scene hand-labeled layouts are merged
    by `label_targets.py` (which has stricter validation requirements).

    Returns the number of scenarios upgraded.
    """
    import csv

    if not csv_path.exists():
        return 0

    by_id = {s.scenario_id: s for s in scenarios}
    upgraded = 0
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = (row.get("scenario_id") or "").strip()
            tgt = (row.get("target_obj_id") or "").strip()
            if not sid or not tgt:
                continue
            s = by_id.get(sid)
            if s is None:
                continue
            if tgt not in [o.id for o in s.objects]:
                continue
            s.target_obj_id = tgt
            s.target_label_source = "hand_label"
            upgraded += 1
    return upgraded


# ── summary report ──────────────────────────────────────────────────────


def _summary(scenarios: List[Scenario], dropped: List[Tuple[str, str]]) -> Dict:
    valid: List[Scenario] = []
    invalid: List[Tuple[str, List[str]]] = []
    for s in scenarios:
        probs = s.validate()
        if probs:
            invalid.append((s.scenario_id, probs))
        else:
            valid.append(s)

    by_mode = Counter()
    by_layout_src = Counter()
    by_target_src = Counter()
    by_subj_diff = Counter()
    for s in valid:
        by_mode[s.source.mode] += 1
        by_layout_src[s.layout_source] += 1
        by_target_src[s.target_label_source] += 1
        by_subj_diff[(s.source.subject, s.source.difficulty)] += 1

    return {
        "total_discovered_trials": len(scenarios) + len(dropped),
        "valid_scenarios": len(valid),
        "dropped_during_extract": [{"trial_dir": d[0], "reason": d[1]} for d in dropped],
        "invalid_after_extract": [{"scenario_id": sid, "problems": probs} for sid, probs in invalid],
        "counts_by_mode": dict(by_mode),
        "counts_by_layout_source": dict(by_layout_src),
        "counts_by_target_source": dict(by_target_src),
        "counts_by_subject_difficulty": {f"{k[0]}/{k[1]}": v for k, v in sorted(by_subj_diff.items())},
        "unlabeled_target_count": by_target_src.get("unlabeled", 0),
    }


# ── main ────────────────────────────────────────────────────────────────


def extract(
    logs_root: Path,
    hand_labels_csv: Optional[Path] = None,
    min_template_support: int = 1,
) -> Tuple[List[Scenario], Dict, List[Tuple[str, str]]]:
    discovered = discover_trials(logs_root)
    trial_data: Dict[Tuple[str, str, str], TrialData] = {}
    dropped: List[Tuple[str, str]] = []

    # Pass 1: load everything. The directory walk is the source of truth for
    # (mode, subject, difficulty); we overwrite any typo'd values in meta /
    # summary (e.g. 7 trials have "assisitive" misspelled in trial_summary).
    by_key: Dict[Tuple[str, str], List[TrialData]] = defaultdict(list)
    for mode, subject, difficulty, trial_dir in discovered:
        td = load_trial(trial_dir)
        if td is None:
            dropped.append((str(trial_dir), "missing trial_summary.json"))
            continue
        td.meta["mode"] = mode
        td.meta["subject_id"] = subject
        td.meta["difficulty"] = difficulty
        by_key[(subject, difficulty)].append(td)

    # Pass 2: build per-(subject, difficulty) canonical templates from assistive trials.
    templates = build_templates(by_key, min_support=min_template_support)
    # Plus per-difficulty fallbacks pooled across subjects, for cells with no
    # assistive data of their own (s1/s2/s5/s6 in our dataset).
    diff_fallbacks = build_difficulty_fallbacks(by_key)

    # Pass 3: emit scenarios.
    scenarios: List[Scenario] = []
    for (subject, difficulty), trials in sorted(by_key.items()):
        per_subj_template = templates.get((subject, difficulty))
        diff_template = diff_fallbacks.get(difficulty)
        for td in trials:
            mode = td.meta.get("mode", "manual")
            if mode == "assistive" and td.tool_calls:
                sc = _scenario_from_assistive_trial(mode, subject, difficulty, td, logs_root)
                if sc is None:
                    dropped.append((str(td.trial_dir), "assistive trial parse failed"))
                    continue
                scenarios.append(sc)
                continue

            # Manual trial OR assistive trial with no tool calls → need template.
            if per_subj_template is not None:
                template = per_subj_template
                note_prefix = f"layout inherited from {template.source_trial_count} assistive {subject}/{difficulty} trials"
            elif diff_template is not None:
                template = diff_template
                note_prefix = (
                    f"layout pooled from {template.source_trial_count} assistive {difficulty} trials "
                    f"across subjects ({template.notes.split(': ', 1)[-1] if template.notes else '?'}); "
                    f"{subject} had no own assistive data"
                )
            else:
                dropped.append(
                    (str(td.trial_dir), f"no scene template for {subject}/{difficulty} (and no difficulty fallback)")
                )
                continue

            sc = _scenario_from_template(mode, subject, difficulty, td, template, logs_root)
            sc.notes = note_prefix
            scenarios.append(sc)

    # Pass 4: optional hand-label overlay.
    if hand_labels_csv is not None:
        n_upgraded = _apply_hand_labels(scenarios, hand_labels_csv)
    else:
        n_upgraded = 0

    summary = _summary(scenarios, dropped)
    summary["hand_label_upgrades"] = n_upgraded
    summary["scene_templates"] = {
        f"{subj}/{diff}": {
            "objects": [(o.label, o.grid_label) for o in tpl.objects],
            "gripper": (tpl.gripper.cell, tpl.gripper.yaw, tpl.gripper.z),
            "n_source_trials": tpl.source_trial_count,
        }
        for (subj, diff), tpl in sorted(templates.items())
    }
    summary["difficulty_fallback_templates"] = {
        diff: {
            "objects": [(o.label, o.grid_label) for o in tpl.objects],
            "gripper": (tpl.gripper.cell, tpl.gripper.yaw, tpl.gripper.z),
            "n_source_trials": tpl.source_trial_count,
            "notes": tpl.notes,
        }
        for diff, tpl in sorted(diff_fallbacks.items())
    }
    return scenarios, summary, dropped


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs_root", default="PRIME_LOGS", help="Root of the PRIME_LOGS directory")
    ap.add_argument(
        "--out_dir",
        default="evaluation/results/robustness/user_input_noise/scenarios",
        help="Output directory for scenarios.jsonl + summary",
    )
    ap.add_argument(
        "--hand_labels",
        default=None,
        help="Optional CSV with target_obj_id overrides per scenario_id",
    )
    ap.add_argument(
        "--min_template_support",
        type=int,
        default=1,
        help="Minimum number of assistive trials supporting an (object, cell) pair "
             "to include it in the canonical scene template",
    )
    args = ap.parse_args(argv)

    logs_root = Path(args.logs_root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hand_csv = Path(args.hand_labels).resolve() if args.hand_labels else None

    scenarios, summary, _dropped = extract(
        logs_root=logs_root,
        hand_labels_csv=hand_csv,
        min_template_support=args.min_template_support,
    )

    valid = [s for s in scenarios if not s.validate()]
    write_scenarios(out_dir / "scenarios.jsonl", valid)
    with (out_dir / "scenarios_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[extract] wrote {len(valid)} valid scenarios to {out_dir / 'scenarios.jsonl'}")
    print(f"[extract] summary at {out_dir / 'scenarios_summary.json'}")
    print(f"[extract] counts by target source: {summary['counts_by_target_source']}")
    print(f"[extract] counts by layout source: {summary['counts_by_layout_source']}")
    print(f"[extract] unlabeled scenarios awaiting hand-labels: {summary['unlabeled_target_count']}")


if __name__ == "__main__":
    main()
