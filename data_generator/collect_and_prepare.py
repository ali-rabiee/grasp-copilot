from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from .episode import write_jsonl
from .generate_dataset import generate as generate_records
from .run_dirs import allocate_numbered_run_dir


def _default_paths(out_dir: str) -> tuple[str, str, str]:
    d = Path(out_dir)
    return (
        str(d / "grasp_gen.jsonl"),
        str(d / "llm_contract.jsonl"),
        str(d / "llm_chat.jsonl"),
    )


def _with_suffix(path: str, suffix: str) -> str:
    p = Path(path)
    if p.suffix == ".jsonl":
        return str(p.with_name(p.stem + suffix + p.suffix))
    return str(p) + suffix


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="One-shot scripted data collection + LLM training data preparation."
    )

    # Collect (generator JSONL)
    ap.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of scripted episodes to generate (required unless --generator_jsonl is provided).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_obj_min", type=int, default=2)
    ap.add_argument("--n_obj_max", type=int, default=10)
    ap.add_argument("--collision_p", type=float, default=0.15)
    ap.add_argument("--candidate_max_dist", type=int, default=1)

    ap.add_argument(
        "--generator_jsonl",
        type=str,
        default=None,
        help="If provided, skip collection and use this generator JSONL as input to preparation.",
    )

    # Outputs
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help=(
            "Directory to write outputs. If omitted, a unique numbered directory is created under "
            "grasp-copilot/data/runs/ (e.g. 001, 002, ...)."
        ),
    )
    ap.add_argument("--out_generator", type=str, default=None, help="Path to write the raw generator JSONL.")
    ap.add_argument("--out_contract", type=str, default=None, help="Path to write dataset-contract JSONL.")
    ap.add_argument("--out_chat", type=str, default=None, help="Path to write chat-formatted JSONL.")
    ap.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Optional override for the dataset instruction prompt (contract format).",
    )

    ap.add_argument(
        "--skip_prepare",
        action="store_true",
        help="Only collect generator JSONL (do not produce contract/chat JSONL).",
    )

    # Optional preprocessing (tool-call rebalance) applied during preparation.
    ap.add_argument(
        "--motion_repeat",
        type=int,
        default=5,
        help="Repeat APPROACH/ALIGN_YAW examples N times in the produced contract JSONL (default: 1 = no rebalance).",
    )
    ap.add_argument(
        "--interact_keep_prob",
        type=float,
        default=1.0,
        help="Keep probability for INTERACT examples in the produced contract JSONL (default: 1.0 = keep all).",
    )
    ap.add_argument("--rebalance_seed", type=int, default=0)

    args = ap.parse_args(argv)

    if args.out_dir:
        out_dir_path = Path(str(args.out_dir))
        out_dir_path.mkdir(parents=True, exist_ok=True)
    else:
        out_dir_path = allocate_numbered_run_dir()
        print(f"[collect] out_dir: {out_dir_path}")

    out_dir = str(out_dir_path)
    out_gen_default, out_contract_default, out_chat_default = _default_paths(out_dir)

    out_generator = str(args.out_generator or out_gen_default)
    out_contract = str(args.out_contract or out_contract_default)
    out_chat = str(args.out_chat or out_chat_default)

    # Ensure parent dirs exist for any requested output.
    for p in (out_generator, out_contract, out_chat):
        os.makedirs(str(Path(p).parent), exist_ok=True)

    if args.generator_jsonl is not None:
        generator_path = str(args.generator_jsonl)
    else:
        if args.episodes is None:
            raise SystemExit("Either provide --episodes (to collect) or --generator_jsonl (to prepare).")

        records, stats = generate_records(
            episodes=int(args.episodes),
            seed=int(args.seed),
            n_obj_min=int(args.n_obj_min),
            n_obj_max=int(args.n_obj_max),
            collision_p=float(args.collision_p),
            candidate_max_dist=int(args.candidate_max_dist),
        )
        write_jsonl(out_generator, records)
        with open(out_generator + ".stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, sort_keys=True)

        generator_path = out_generator

    if args.skip_prepare:
        return

    # Keep the LLM preparation logic in llm.*; imported lazily so data_generator can be
    # used without pulling in training dependencies.
    from llm.data import convert_contract_to_qwen_chat_jsonl, convert_generator_jsonl_to_contract
    from llm.rebalance_contract import rebalance_contract

    convert_generator_jsonl_to_contract(
        generator_path=generator_path,
        out_path=out_contract,
        instruction=args.instruction,
        max_past_dialogs=12,
    )

    if int(args.motion_repeat) != 1 or float(args.interact_keep_prob) != 1.0:
        # Keep the original contract for evaluation/debugging, and write a separate rebalanced contract for training.
        out_contract_reb = _with_suffix(out_contract, "_rebalanced")
        tmp_out = str(out_contract_reb) + ".tmp_rebalanced"
        stats = rebalance_contract(
            in_path=str(out_contract),
            out_path=tmp_out,
            seed=int(args.rebalance_seed),
            motion_repeat=int(args.motion_repeat),
            interact_keep_prob=float(args.interact_keep_prob),
        )
        os.replace(tmp_out, str(out_contract_reb))
        print(f"[collect] rebalanced contract written to {out_contract_reb} | stats={stats}")

        out_chat_reb = _with_suffix(out_chat, "_rebalanced")
        convert_contract_to_qwen_chat_jsonl(out_contract_reb, out_chat_reb)
        print(f"[collect] rebalanced chat written to {out_chat_reb}")

    convert_contract_to_qwen_chat_jsonl(out_contract, out_chat)


if __name__ == "__main__":
    main()


