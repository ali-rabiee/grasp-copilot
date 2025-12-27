from __future__ import annotations

import argparse
from typing import Optional

from .data import convert_contract_to_qwen_chat_jsonl, convert_generator_jsonl_to_contract


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
    args = ap.parse_args(argv)

    convert_generator_jsonl_to_contract(args.generator_jsonl, args.out_contract, max_past_dialogs=int(args.max_past_dialogs))
    convert_contract_to_qwen_chat_jsonl(args.out_contract, args.out_chat)


if __name__ == "__main__":
    main()


