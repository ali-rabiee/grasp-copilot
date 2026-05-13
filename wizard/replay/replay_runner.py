"""Replay-mode driver for the wizard.

Instead of generating fresh schematic episodes, the wizard annotates real
oracle-rolled-out states sampled from existing ``grasp_gen*.jsonl`` files.

Why this exists:

* The schematic env at the wizard's session start has empty ``past_dialogs``,
  empty ``last_prompt``, and freshly-warmed gripper history — i.e., very
  little context. Most decisions land in the same "first-tick of a sterile
  episode" pose, which doesn't match what the deployed LLM will see at
  inference time.
* The oracle's data already contains thousands of *realistic* mid-episode
  states with rich memory, real dialog history, and pruned candidate sets —
  exactly the distribution the model needs to handle.
* Annotating those states (state-only; the wizard never sees the oracle's
  choice) gives high-quality WoZ labels that line up 1:1 with the inference
  distribution.

This driver has the same surface as ``EpisodeRunner.run()`` so ``WizardApp``
can use either interchangeably.
"""

from __future__ import annotations

import json
import random as _random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..driver.alert_scheduler import AlertReason
from ..io.writer import EpisodeWriter

DecisionFn = Callable[[Dict, AlertReason], Dict]


@dataclass
class ReplayConfig:
    """How to choose which records to surface to the wizard.

    Default mode is **episode-coherent**: records are replayed in source
    order, grouped by episode_id. Motion records (APPROACH/STACK/POUR/…) are
    auto-applied with the oracle's tool call so the wizard sees an episode
    unfold naturally and only has to make decisions at INTERACT alerts.
    This avoids the "cold context" problem of jumping into the middle of an
    episode with 10+ prior dialog turns.
    """

    replay_jsonl: Path
    env_name: str
    wizard_id: str = "anon"
    # Block on INTERACT records; auto-apply motion records.
    block_only_on_interact: bool = True
    """If True (default), motion records (APPROACH, STACK, etc.) are
    auto-logged using the oracle's tool call without bothering the wizard.
    Only INTERACT records surface to the GUI. If False, the wizard is asked
    at every record."""
    # Replay ordering — defaults to "episode-coherent" (no shuffle).
    randomize: bool = False
    seed: int = 0
    # Source filtering / partitioning
    max_records: Optional[int] = None
    skip_records: int = 0
    max_episodes: Optional[int] = None
    """If set, stop after this many distinct source episodes have been fully
    replayed. Useful for the 'annotate the first 100 episodes' workflow."""
    skip_episodes: int = 0
    """Skip the first N source episodes before starting. Useful for wizard
    partitioning: alice does eps 0-99, bob does eps 100-199, etc."""


class ReplayEnvShim:
    """Stand-in for ``SchematicEnv`` so ``WizardApp`` can read the same
    attributes (``episode_idx``, ``tick``, ``cfg.env_name``)."""

    @dataclass
    class _Cfg:
        env_name: str = "replay"

    def __init__(self, env_name: str):
        self.episode_idx: int = -1
        self.tick: int = 0
        self.cfg = ReplayEnvShim._Cfg(env_name=env_name)


class ReplayRunner:
    """Surface pre-collected oracle records to the wizard one at a time."""

    def __init__(self, cfg: ReplayConfig, writer: EpisodeWriter, decision_fn: DecisionFn):
        self.cfg = cfg
        self.writer = writer
        self.decide = decision_fn
        self.env = ReplayEnvShim(env_name=cfg.env_name)
        self._records: Optional[List[Dict]] = None

    # ------------------------------------------------------------------ load

    def _load_records(self) -> List[Dict]:
        """Load and order source records.

        Default ordering: episode-coherent (source order, grouped by
        episode_id). With ``randomize=True``, episodes themselves are shuffled
        but records inside an episode stay in their original order — so the
        wizard never sees a mid-episode record before its prefix.
        """
        if self._records is not None:
            return self._records
        path = Path(self.cfg.replay_jsonl)
        raw: List[Dict] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec_env = rec.get("env")
                if rec_env and rec_env != self.cfg.env_name:
                    continue
                raw.append(rec)

        # Group by episode_id, preserving source order inside each episode.
        from collections import OrderedDict
        groups: "OrderedDict[int, List[Dict]]" = OrderedDict()
        for rec in raw:
            ep_id = int(rec.get("episode_id", 0))
            groups.setdefault(ep_id, []).append(rec)
        episode_order = list(groups.keys())
        if self.cfg.randomize:
            _random.Random(self.cfg.seed).shuffle(episode_order)
        if self.cfg.skip_episodes:
            episode_order = episode_order[int(self.cfg.skip_episodes):]
        if self.cfg.max_episodes is not None:
            episode_order = episode_order[: int(self.cfg.max_episodes)]

        # Flatten back to a single record stream in the chosen episode order.
        records: List[Dict] = []
        for ep_id in episode_order:
            records.extend(groups[ep_id])

        if self.cfg.skip_records:
            records = records[int(self.cfg.skip_records):]
        if self.cfg.max_records is not None:
            records = records[: int(self.cfg.max_records)]
        self._records = records
        return records

    def num_records(self) -> int:
        return len(self._load_records())

    def num_blocking_records(self) -> int:
        """Number of records that will actually require wizard input."""
        records = self._load_records()
        if not self.cfg.block_only_on_interact:
            return len(records)
        return sum(1 for r in records if r["target_tool_call"]["tool"] == "INTERACT")

    # ------------------------------------------------------------------- run

    def run_episode(self) -> int:
        """Walk the source records in episode-coherent order.

        For each record:
          * If the oracle emitted INTERACT, surface to the wizard via the
            decision callback (blocking on GUI input).
          * Otherwise, auto-log the oracle's motion tool call (APPROACH,
            STACK, POUR, …) without bothering the wizard.

        Either way the record is written to the wizard's JSONL — so the
        resulting dataset contains a complete episode trajectory, mixing
        wizard-authored INTERACTs with oracle-derived motion calls. The
        wizard's session only blocks at decision points, which is the whole
        point of replay mode.
        """
        records = self._load_records()
        if not records:
            return 0

        n_done = 0
        last_source_ep_id: Optional[int] = None

        for rec in records:
            source_ep_id = int(rec.get("episode_id", n_done))

            # New source episode → close previous, open new one.
            if source_ep_id != last_source_ep_id:
                if last_source_ep_id is not None:
                    self.writer.end_episode()
                self.env.episode_idx += 1
                self.env.tick = 0
                self.writer.start_episode(
                    episode_id=self.env.episode_idx,
                    env_name=self.cfg.env_name,
                    wizard_id=self.cfg.wizard_id,
                    intended_obj_id=rec.get("intended_obj_id", ""),
                )
                last_source_ep_id = source_ep_id
            else:
                self.env.tick += 1

            blob = {
                "objects": rec["objects"],
                "gripper_hist": rec["gripper_hist"],
                "memory": rec["memory"],
                "user_state": rec.get("user_state", {"mode": "translation"}),
            }

            oracle_tc = rec["target_tool_call"]
            is_decision_point = oracle_tc["tool"] == "INTERACT"

            if is_decision_point or not self.cfg.block_only_on_interact:
                # Block for wizard input.
                tool_call = self.decide(blob, AlertReason.REPLAY)
                alert_reason = AlertReason.REPLAY.value
            else:
                # Motion record: auto-apply the oracle's tool call. The wizard
                # is not interrupted — but the record is still logged so the
                # resulting dataset contains a complete episode trajectory.
                tool_call = oracle_tc
                alert_reason = "auto_motion"

            self.writer.write_decision(
                blob=blob,
                tool_call=tool_call,
                alert_reason=alert_reason,
            )
            n_done += 1

        if last_source_ep_id is not None:
            self.writer.end_episode()
        return n_done
