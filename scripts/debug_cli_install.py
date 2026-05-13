#!/usr/bin/env python3
"""Debug helper: log why `grasp-collect` may be missing from PATH (session NDJSON)."""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

_LOG = Path("/home/ali/github/ali-rabiee/.cursor/debug-187af4.log")
_SESSION = "187af4"


def _log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # #region agent log
    line = json.dumps(
        {
            "sessionId": _SESSION,
            "runId": "cli-diagnose",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        },
        ensure_ascii=False,
    )
    _LOG.parent.mkdir(parents=True, exist_ok=True)
    with _LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    # #endregion


def main() -> int:
    which = shutil.which("grasp-collect")
    _log(
        "A",
        "scripts/debug_cli_install.py:which",
        "grasp-collect on PATH",
        {"which": which, "sys.executable": sys.executable},
    )

    dist_ver = None
    dist_ok = False
    try:
        from importlib.metadata import distribution

        d = distribution("grasp-copilot")
        dist_ok = True
        dist_ver = d.version
    except Exception as e:  # noqa: BLE001
        _log(
            "B",
            "scripts/debug_cli_install.py:metadata",
            "importlib.metadata distribution('grasp-copilot')",
            {"ok": False, "error": repr(e)},
        )
    else:
        eps = []
        try:
            eps = [f"{ep.name}={ep.value}" for ep in d.entry_points if ep.group == "console_scripts"]
        except Exception as e:  # noqa: BLE001
            eps = [f"<list_failed:{e!r}>"]
        _log(
            "B",
            "scripts/debug_cli_install.py:metadata",
            "importlib.metadata distribution('grasp-copilot')",
            {"ok": True, "version": dist_ver, "console_scripts_sample": eps[:12]},
        )

    try:
        import data_generator.collect_and_prepare as cap  # noqa: PLC0415

        mod_ok = True
        mod_file = str(getattr(cap, "__file__", ""))
    except Exception as e:  # noqa: BLE001
        mod_ok = False
        mod_file = repr(e)
    _log(
        "C",
        "scripts/debug_cli_install.py:import",
        "import data_generator.collect_and_prepare",
        {"ok": mod_ok, "module_file": mod_file, "cwd": str(Path.cwd())},
    )

    print("Wrote diagnostics to", _LOG)
    print("grasp-collect which:", which)
    print("grasp-copilot installed:", dist_ok, dist_ver or "")
    print("If which is None: run from grasp-copilot:  pip install -e .")
    print("Alternate run without console script:")
    print("  python -m data_generator.collect_and_prepare --help")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
