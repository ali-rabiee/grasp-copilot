import argparse

import _bootstrap  # noqa: F401
from llm.prepare_llm_data import main as _main


def main() -> None:
    # Backward-compatible shim. Prefer:
    #   python -m llm.prepare_llm_data ...
    _main()


if __name__ == "__main__":
    main()

