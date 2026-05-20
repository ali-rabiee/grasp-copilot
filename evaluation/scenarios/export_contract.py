"""
Export scenarios.jsonl as contract JSONL — the format every evaluator in
`evaluation/` (offline_exec_benchmark, robustness_benchmark, run_full_benchmark)
already reads.

Usage:
    # Export every scenario, with empty `output` (inference-only):
    python -m evaluation.scenarios.export_contract \\
        --scenarios evaluation/eval_outputs/scenario_noise/scenarios.jsonl \\
        --out evaluation/eval_outputs/scenario_noise/scenarios_contract.jsonl

    # Export only labeled scenarios + fill `output` with the heuristic oracle:
    python -m evaluation.scenarios.export_contract \\
        --scenarios evaluation/eval_outputs/scenario_noise/scenarios.labeled.jsonl \\
        --out evaluation/eval_outputs/scenario_noise/scenarios_contract.jsonl \\
        --skip_unlabeled --with_oracle_output

    # Then it's directly consumable by:
    python -m evaluation.offline_exec_benchmark \\
        --contract_jsonl evaluation/eval_outputs/scenario_noise/scenarios_contract.jsonl \\
        --models "Qwen7B-FT=models/qwen2_5_7b_instruct_ft"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List

try:
    from evaluation import _bootstrap  # noqa: F401
except Exception:
    import _bootstrap  # type: ignore  # noqa: F401

from evaluation.scenarios import load_scenarios
from evaluation.scenarios.adapters import (
    DEFAULT_CANDIDATE_MAX_DIST,
    DEFAULT_INSTRUCTION,
    DEFAULT_USER_MODE,
    oracle_output_for,
    write_scenarios_as_contract_jsonl,
)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", required=True, help="Input scenarios.jsonl")
    ap.add_argument("--out", required=True, help="Output contract JSONL path")
    ap.add_argument(
        "--skip_unlabeled",
        action="store_true",
        help="Drop scenarios whose target_obj_id is None",
    )
    ap.add_argument(
        "--with_oracle_output",
        action="store_true",
        help="Run the heuristic oracle on each scenario to fill the `output` field "
             "(requires target_obj_id to be set)",
    )
    ap.add_argument(
        "--user_mode",
        default=DEFAULT_USER_MODE,
        choices=("translation", "rotation", "gripper"),
        help="Initial user_state.mode written into every contract row",
    )
    ap.add_argument(
        "--candidate_max_dist",
        type=int,
        default=DEFAULT_CANDIDATE_MAX_DIST,
        help="Manhattan radius for the t=0 candidate set",
    )
    ap.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
        help="Instruction string written into every row",
    )
    args = ap.parse_args(argv)

    scenarios = load_scenarios(args.scenarios)
    out_path = Path(args.out)

    oracle_fn = oracle_output_for if args.with_oracle_output else None

    n_written = write_scenarios_as_contract_jsonl(
        scenarios,
        out_path,
        instruction=args.instruction,
        user_mode=args.user_mode,
        candidate_max_dist=args.candidate_max_dist,
        skip_unlabeled=args.skip_unlabeled,
        output_tool_call_fn=oracle_fn,
    )

    print(f"[export] read {len(scenarios)} scenarios from {args.scenarios}")
    print(f"[export] wrote {n_written} contract rows to {out_path}")
    if args.skip_unlabeled and n_written < len(scenarios):
        print(f"[export] skipped {len(scenarios) - n_written} unlabeled scenarios")


if __name__ == "__main__":
    main()
