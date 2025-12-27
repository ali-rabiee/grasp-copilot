from __future__ import annotations

import argparse
from typing import Optional

from .data import convert_contract_to_qwen_chat_jsonl, convert_generator_jsonl_to_contract
from .rebalance_contract import rebalance_contract


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Convert generator JSONL into contract + chat JSONL for training.")
    ap.add_argument("--generator_jsonl", type=str, required=True)
    ap.add_argument("--out_contract", type=str, required=True)
    ap.add_argument("--out_chat", type=str, required=True)
    ap.add_argument(
        "--max_past_dialogs",
        type=int,
        default=12,
        help="Keep only the last N dialog messages in memory.past_dialogs to reduce prompt truncation.",
    )
    ap.add_argument(
        "--motion_repeat",
        type=int,
        default=1,
        help="Optional preprocessing: repeat APPROACH/ALIGN_YAW examples N times to reduce motion->INTERACT bias.",
    )
    ap.add_argument(
        "--interact_keep_prob",
        type=float,
        default=1.0,
        help="Optional preprocessing: keep probability for INTERACT examples (1.0 keeps all, <1.0 downsamples).",
    )
    ap.add_argument("--rebalance_seed", type=int, default=0)
    args = ap.parse_args(argv)

    # Step 1: generator -> contract
    convert_generator_jsonl_to_contract(args.generator_jsonl, args.out_contract, max_past_dialogs=int(args.max_past_dialogs))

    # Step 2 (optional): rebalance tool-call frequencies *as a preprocessing step*
    if int(args.motion_repeat) != 1 or float(args.interact_keep_prob) != 1.0:
        # Keep original contract and also write a separate rebalanced one.
        out_contract_reb = str(args.out_contract) + ".rebalanced"
        tmp_out = out_contract_reb + ".tmp"
        stats = rebalance_contract(
            in_path=str(args.out_contract),
            out_path=tmp_out,
            seed=int(args.rebalance_seed),
            motion_repeat=int(args.motion_repeat),
            interact_keep_prob=float(args.interact_keep_prob),
        )
        import os

        os.replace(tmp_out, out_contract_reb)
        print(f"[prepare] rebalanced contract written to {out_contract_reb} | stats={stats}")

        # Write matching rebalanced chat file too.
        out_chat_reb = str(args.out_chat) + ".rebalanced"
        convert_contract_to_qwen_chat_jsonl(out_contract_reb, out_chat_reb)
        print(f"[prepare] rebalanced chat written to {out_chat_reb}")

    # Step 3: contract -> chat
    convert_contract_to_qwen_chat_jsonl(args.out_contract, args.out_chat)


if __name__ == "__main__":
    main()


