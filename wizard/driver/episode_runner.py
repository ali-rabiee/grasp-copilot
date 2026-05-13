"""Episode loop that connects schematic env, simulated user, alert scheduler,
the wizard GUI, and the JSONL writer.

The runner is GUI-agnostic. ``WizardApp`` injects its blocking
``request_decision`` callback so the loop can pause for human input without
the runner needing to know about Tkinter.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ..env.schematic_env import EnvConfig, SchematicEnv
from ..io.writer import EpisodeWriter
from .alert_scheduler import AlertReason, AlertScheduler
from .user_model import SimulatedUser, UserConfig

# Type of the wizard's blocking decision callback.
#   Input  → the symbolic state blob + alert reason
#   Output → a validated PRIME tool call dict {"tool": ..., "args": ...}
DecisionFn = Callable[[Dict, AlertReason], Dict]


@dataclass
class RunnerConfig:
    env_cfg: EnvConfig = field(default_factory=EnvConfig)
    user_cfg: UserConfig = field(default_factory=UserConfig)
    p_alert: float = 0.15
    max_ticks_per_episode: int = 200
    max_alerts_per_episode: int = 25
    wizard_id: str = "anon"
    seed: Optional[int] = None


class EpisodeRunner:
    """Drive a single episode at a time. The GUI calls ``run_episode`` per episode."""

    def __init__(self, cfg: RunnerConfig, writer: EpisodeWriter, decision_fn: DecisionFn):
        self.cfg = cfg
        self.writer = writer
        self.decide = decision_fn
        self.rng = random.Random(cfg.seed)
        env_cfg = cfg.env_cfg
        if env_cfg.seed is None:
            env_cfg.seed = self.rng.randrange(1 << 30)
        user_cfg = cfg.user_cfg
        if user_cfg.seed is None:
            user_cfg.seed = self.rng.randrange(1 << 30)
        self.env = SchematicEnv(env_cfg)
        self.user = SimulatedUser(user_cfg)
        self.scheduler = AlertScheduler(p_alert=cfg.p_alert, seed=self.rng.randrange(1 << 30))

    # ------------------------------------------------------------------ run

    def run_episode(self) -> int:
        """Run one episode, returning the number of decisions logged."""
        self.env.reset()
        self.scheduler.reset_episode()
        self.writer.start_episode(
            episode_id=self.env.episode_idx,
            env_name=self.env.cfg.env_name,
            wizard_id=self.cfg.wizard_id,
            intended_obj_id=self.env.intended_obj_id,
        )

        decisions = 0
        terminate = False

        for _ in range(self.cfg.max_ticks_per_episode):
            blob = self.env.public_blob()
            reason = self.scheduler.should_alert(blob["memory"]["candidates"])

            if reason is not None and decisions < self.cfg.max_alerts_per_episode:
                tool_call = self.decide(blob, reason)
                self.writer.write_decision(blob=blob, tool_call=tool_call, alert_reason=reason.value)
                decisions += 1

                terminate = self._apply_wizard_decision(tool_call)
                if terminate:
                    break
                continue

            cmd = self.user.step(self.env)
            self.env.apply_user_command(cmd)

            target = next((o for o in self.env.objects if o.id == self.env.intended_obj_id), None)
            if target is not None and self.env.gripper.cell == target.cell and self.env.gripper.yaw == target.yaw:
                break

        self.writer.end_episode()
        return decisions

    # -------------------------------------------------- decision dispatcher

    def _apply_wizard_decision(self, tool_call: Dict) -> bool:
        """Apply a wizard tool call to the env. Returns True iff episode should end."""
        tool = tool_call.get("tool")
        args = tool_call.get("args") or {}

        if tool == "INTERACT":
            kind = args.get("kind", "QUESTION")
            text = args.get("text", "")
            choices = list(args.get("choices") or [])
            self.env.memory["last_prompt"] = {"kind": kind, "text": text, "choices": choices}
            reply = self.user.reply_to_prompt(self.env, kind, choices)
            self.env.apply_interaction(kind, text, choices, reply)
            self._update_candidates_from_reply(kind, text, choices, reply)
            return False

        if tool in {"APPROACH", "ALIGN_YAW", "STACK", "RELEASE", "GRAB", "POUR"}:
            obj_id = args.get("obj", "")
            self.env.apply_execution_skill(tool, obj_id, args.get("amount"))
            self.scheduler.mark_post_execution()
            target = next((o for o in self.env.objects if o.id == self.env.intended_obj_id), None)
            if target is not None and obj_id == target.id and tool in {"ALIGN_YAW", "STACK", "POUR"}:
                return True
            return False

        return False

    def _update_candidates_from_reply(self, kind: str, text: str, choices: List[str], reply: str) -> None:
        """Mirror the oracle's deterministic candidate-pruning rule.

        If the prompt names specific object labels and the user picks one,
        treat the others as excluded. If reply is "NO", exclude any single
        candidate the prompt referenced.
        """
        upper = reply.upper()
        objs = self.env.objects

        for o in objs:
            if o.label.lower() in reply.lower() and o.label.lower() not in text.lower().split(reply.lower())[0:1]:
                pass  # placeholder — handled below

        for c in choices:
            for o in objs:
                if o.label.lower() in c.lower() and c == reply:
                    for other in objs:
                        if other.id != o.id and other.label.lower() in text.lower():
                            self.env.exclude_obj(other.id)
                    return

        if "NO" in upper:
            for o in objs:
                if o.label.lower() in text.lower():
                    self.env.exclude_obj(o.id)
                    break
