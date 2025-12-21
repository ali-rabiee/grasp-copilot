from __future__ import annotations

import argparse
from typing import Optional

from .data import convert_contract_to_qwen_chat_jsonl, convert_generator_jsonl_to_contract


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Convert generator JSONL into contract + chat JSONL for training.")
    ap.add_argument("--generator_jsonl", type=str, required=True)
    ap.add_argument("--out_contract", type=str, required=True)
    ap.add_argument("--out_chat", type=str, required=True)
    args = ap.parse_args(argv)

    convert_generator_jsonl_to_contract(args.generator_jsonl, args.out_contract)
    convert_contract_to_qwen_chat_jsonl(args.out_contract, args.out_chat)


if __name__ == "__main__":
    main()


