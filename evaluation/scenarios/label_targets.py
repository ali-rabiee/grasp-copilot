"""
Hand-label workflow for scenarios that the extractor could not auto-label.

Two CLI verbs:

    # Generate a pre-populated CSV stub from the current scenarios.jsonl.
    python -m evaluation.scenarios.label_targets stub \\
        --scenarios evaluation/eval_outputs/scenario_noise/scenarios.jsonl \\
        --out evaluation/scenarios/manual_targets.csv

    # Merge a human-edited CSV back into scenarios.jsonl.
    python -m evaluation.scenarios.label_targets merge \\
        --scenarios evaluation/eval_outputs/scenario_noise/scenarios.jsonl \\
        --csv evaluation/scenarios/manual_targets.csv \\
        --out evaluation/eval_outputs/scenario_noise/scenarios.labeled.jsonl

The CSV is the source of truth for `target_obj_id` (and, optionally, for the
whole object layout when the borrowed template is wrong). Provenance gets
upgraded to `hand_label` for any field the CSV overrides.

CSV columns:
    scenario_id          required, must match scenarios.jsonl
    target_obj_id        e.g. "obj_3"; must match one of the listed object ids
    plausible_targets    "|"-joined, auto-populated for convenience (read-only)
    object_summary       "obj_1=mug@A2|obj_2=cleanser@A3|..." (read-only)
    layout_override      optional JSON string with {objects:[...], gripper_init:{...}}
                         only set if the borrowed scene template is wrong for this
                         specific trial; leave empty in the common case
    notes                optional free-text
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional

try:
    from evaluation import _bootstrap  # noqa: F401
except Exception:
    import _bootstrap  # type: ignore  # noqa: F401

from evaluation.scenarios.schema import (
    LAYOUT_SOURCE_HAND_LABEL,
    TARGET_SOURCE_HAND_LABEL,
    TARGET_SOURCE_UNLABELED,
    GripperInit,
    ObjectInit,
    Scenario,
    load_scenarios,
    write_scenarios,
)


CSV_FIELDS = [
    "scenario_id",
    "subject",
    "difficulty",
    "mode",
    "target_obj_id",
    "plausible_targets",
    "object_summary",
    "current_target_source",
    "layout_override",
    "notes",
]


def _object_summary(s: Scenario) -> str:
    return "|".join(f"{o.id}={o.label}@{o.cell}({o.yaw})" for o in s.objects)


def stub(scenarios_path: Path, csv_out: Path, only_unlabeled: bool = True) -> int:
    scenarios = load_scenarios(scenarios_path)
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in scenarios:
        if only_unlabeled and s.target_label_source != TARGET_SOURCE_UNLABELED:
            continue
        rows.append({
            "scenario_id": s.scenario_id,
            "subject": s.source.subject,
            "difficulty": s.source.difficulty,
            "mode": s.source.mode,
            "target_obj_id": "",  # human fills this in
            "plausible_targets": "|".join(s.plausible_target_ids),
            "object_summary": _object_summary(s),
            "current_target_source": s.target_label_source,
            "layout_override": "",
            "notes": "",
        })

    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[stub] wrote {len(rows)} rows to {csv_out}")
    return len(rows)


def _parse_layout_override(raw: str) -> Optional[dict]:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"layout_override is not valid JSON: {exc}") from exc


def merge(scenarios_path: Path, csv_path: Path, out_path: Path) -> dict:
    scenarios = load_scenarios(scenarios_path)
    by_id = {s.scenario_id: s for s in scenarios}

    n_target = 0
    n_layout = 0
    skipped: List[str] = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = (row.get("scenario_id") or "").strip()
            if not sid:
                continue
            s = by_id.get(sid)
            if s is None:
                skipped.append(f"{sid}: not found in scenarios.jsonl")
                continue

            # Layout override (rare).
            layout = _parse_layout_override(row.get("layout_override") or "")
            if layout is not None:
                try:
                    s.objects = [ObjectInit(**o) for o in layout["objects"]]
                    s.gripper_init = GripperInit(**layout["gripper_init"])
                    s.layout_source = LAYOUT_SOURCE_HAND_LABEL
                    n_layout += 1
                except Exception as exc:
                    skipped.append(f"{sid}: layout_override malformed: {exc}")
                    continue

            # Target.
            tgt = (row.get("target_obj_id") or "").strip()
            if tgt:
                if tgt not in [o.id for o in s.objects]:
                    skipped.append(
                        f"{sid}: target {tgt} not in current objects {[o.id for o in s.objects]}"
                    )
                    continue
                s.target_obj_id = tgt
                s.target_label_source = TARGET_SOURCE_HAND_LABEL
                n_target += 1

            note = (row.get("notes") or "").strip()
            if note:
                s.notes = (s.notes + " | " + note).strip(" |")

    # Validate every scenario and drop invalid ones with a loud report.
    valid: List[Scenario] = []
    invalid: List[str] = []
    for s in scenarios:
        probs = s.validate()
        if probs:
            invalid.append(f"{s.scenario_id}: {probs}")
        else:
            valid.append(s)

    write_scenarios(out_path, valid)

    report = {
        "n_scenarios_total": len(scenarios),
        "n_targets_upgraded": n_target,
        "n_layouts_overridden": n_layout,
        "n_rows_skipped": len(skipped),
        "skipped_reasons": skipped[:20],   # cap to keep the report readable
        "n_invalid_after_merge": len(invalid),
        "invalid_examples": invalid[:20],
        "output": str(out_path),
    }
    print(json.dumps(report, indent=2))
    return report


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_stub = sub.add_parser("stub", help="Generate a CSV stub")
    p_stub.add_argument("--scenarios", required=True)
    p_stub.add_argument("--out", required=True)
    p_stub.add_argument(
        "--include_labeled",
        action="store_true",
        help="Also include scenarios that already have an auto-labeled target",
    )

    p_merge = sub.add_parser("merge", help="Merge a human-edited CSV into scenarios.jsonl")
    p_merge.add_argument("--scenarios", required=True)
    p_merge.add_argument("--csv", required=True)
    p_merge.add_argument("--out", required=True)

    args = ap.parse_args(argv)
    if args.cmd == "stub":
        stub(Path(args.scenarios), Path(args.out), only_unlabeled=not args.include_labeled)
    elif args.cmd == "merge":
        merge(Path(args.scenarios), Path(args.csv), Path(args.out))


if __name__ == "__main__":
    main()
