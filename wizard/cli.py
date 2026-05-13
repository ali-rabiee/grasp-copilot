"""Command-line entry point for the wizard package.

Subcommands::

    python -m wizard collect --env reach_to_grasp_ycb --num-episodes 50 \
        --wizard-id alice --output grasp-copilot/wizard/data/runs/woz_001

    python -m wizard kappa --agreement-dir <run_dir>/agreement

The collect subcommand brings up the GUI and runs ``num_episodes`` episodes;
kappa wraps ``analysis.kappa.main``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .driver.episode_runner import RunnerConfig
from .driver.user_model import UserConfig
from .env.schematic_env import ENVS, EnvConfig
from .io.writer import EpisodeWriter


def _parse_collect(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="wizard collect")
    parser.add_argument("--env", choices=list(ENVS), default="reach_to_grasp_ycb")
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="(live mode only) Number of fresh schematic episodes to drive.")
    parser.add_argument("--wizard-id", required=True)
    parser.add_argument("--output", required=True,
                        help="Run directory to append into (will be created).")
    parser.add_argument("--p-alert", type=float, default=0.15)
    parser.add_argument("--max-alerts", type=int, default=25)
    parser.add_argument("--max-ticks", type=int, default=200)
    parser.add_argument("--n-objects-min", type=int, default=2)
    parser.add_argument("--n-objects-max", type=int, default=6)
    parser.add_argument("--seed", type=int, default=None)

    # Replay-from-real-data mode (recommended for WoZ data collection).
    parser.add_argument(
        "--replay-jsonl",
        type=str,
        default=None,
        help="Path to a grasp_gen*.jsonl from data/. When provided, the wizard "
             "annotates pre-collected oracle states instead of driving fresh "
             "schematic episodes. The state shown to the wizard (memory, dialog "
             "history, gripper history) is the oracle's real rollout context.",
    )
    parser.add_argument(
        "--replay-block-only-on-interact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="(default: on) Walk the episode in source order, auto-applying "
             "motion records and only blocking for wizard input at INTERACT "
             "decision points. Turn off to ask the wizard at every record.",
    )
    parser.add_argument(
        "--replay-randomize",
        action="store_true",
        help="Shuffle EPISODES (not individual records). Records inside an "
             "episode stay in source order — the wizard never sees mid-episode "
             "context without its prefix.",
    )
    parser.add_argument("--replay-max-episodes", type=int, default=None,
                        help="Cap the number of source episodes to annotate.")
    parser.add_argument("--replay-skip-episodes", type=int, default=0,
                        help="Skip the first N episodes (multi-wizard partitioning).")
    parser.add_argument("--replay-max-records", type=int, default=None,
                        help="Cap the total record count after the episode-level cap.")
    parser.add_argument("--replay-skip-records", type=int, default=0,
                        help="Skip the first N records after episode-level filtering.")
    return parser.parse_args(args)


def _cmd_collect(args: List[str]) -> None:
    ns = _parse_collect(args)
    import tkinter as tk
    from .gui.app import WizardApp

    out_dir = Path(ns.output)
    writer = EpisodeWriter(out_dir=out_dir, wizard_id=ns.wizard_id)

    if ns.replay_jsonl:
        # Replay-from-real-data mode: surface pre-collected oracle states.
        from .replay import ReplayConfig, ReplayRunner
        replay_cfg = ReplayConfig(
            replay_jsonl=Path(ns.replay_jsonl),
            env_name=ns.env,
            wizard_id=ns.wizard_id,
            block_only_on_interact=bool(ns.replay_block_only_on_interact),
            randomize=bool(ns.replay_randomize),
            seed=int(ns.seed) if ns.seed is not None else 0,
            max_records=ns.replay_max_records,
            skip_records=int(ns.replay_skip_records),
            max_episodes=ns.replay_max_episodes,
            skip_episodes=int(ns.replay_skip_episodes),
        )
        root = tk.Tk()
        WizardApp(root, runner_cfg=None, writer=writer, num_episodes=1,
                  replay_cfg=replay_cfg)
        root.mainloop()
        return

    # Live mode: drive fresh schematic episodes.
    runner_cfg = RunnerConfig(
        env_cfg=EnvConfig(
            env_name=ns.env,
            n_objects_min=ns.n_objects_min,
            n_objects_max=ns.n_objects_max,
            seed=ns.seed,
        ),
        user_cfg=UserConfig(seed=ns.seed),
        p_alert=ns.p_alert,
        max_ticks_per_episode=ns.max_ticks,
        max_alerts_per_episode=ns.max_alerts,
        wizard_id=ns.wizard_id,
        seed=ns.seed,
    )
    root = tk.Tk()
    WizardApp(root, runner_cfg=runner_cfg, writer=writer, num_episodes=ns.num_episodes)
    root.mainloop()


def _cmd_kappa(argv: List[str]) -> None:
    from .analysis import kappa as kappa_mod
    sys.argv = ["wizard kappa", *argv]
    kappa_mod.main()


def main(argv: Optional[List[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("Usage: python -m wizard {collect,kappa} ...")
        sys.exit(2)
    sub, rest = argv[0], argv[1:]
    if sub == "collect":
        _cmd_collect(rest)
    elif sub == "kappa":
        _cmd_kappa(rest)
    else:
        print(f"Unknown subcommand: {sub}")
        sys.exit(2)


if __name__ == "__main__":
    main()
