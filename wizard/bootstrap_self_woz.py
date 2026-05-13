"""Bootstrap self-WoZ data in the PRIME generator schema.

This is a deterministic, scriptable stand-in for an initial Wizard-of-Oz pass.
It uses the same prompt templates as the oracle-facing GUI and writes:

    data/woz_<env>/grasp_gen.jsonl
    data/woz_<env>/episodes_meta.jsonl
    data/woz_<env>/summary.json

The policy is intentionally conservative:
  1. Ask a candidate-choice question while the candidate set is ambiguous.
  2. Prune the memory to the user's selected target.
  3. Confirm any state-changing action before executing it.
  4. Add a follow-up assistive prompt before the next action or closure.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from data_generator.oracle import validate_tool_call

from .env.schematic_env import ENVS, EnvConfig, SchematicEnv
from .prompt_factory import build_prompt


DEFAULT_SAMPLES_PER_ENV = 500
RECORDS_PER_EPISODE = 5
WIZARD_ID = "self_woz_bootstrap"


def _obj_by_id(blob: Dict[str, Any], obj_id: str) -> Dict[str, Any]:
    for obj in blob.get("objects") or []:
        if obj.get("id") == obj_id:
            return obj
    raise KeyError(obj_id)


def _objects_for_env(env: str) -> Tuple[int, int]:
    if env == "reach_to_grasp_ycb":
        return 5, 6
    if env == "cube_stacking":
        return 3, 4
    return 3, 3


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class SelfWozCollector:
    def __init__(self, env_name: str, *, samples: int, seed: int, output_root: Path):
        if samples % RECORDS_PER_EPISODE != 0:
            raise ValueError(f"samples must be divisible by {RECORDS_PER_EPISODE}")
        self.env_name = env_name
        self.samples = samples
        self.seed = seed
        self.rng = random.Random(seed)
        self.output_dir = output_root / f"woz_{env_name}"
        n_min, n_max = _objects_for_env(env_name)
        self.env = SchematicEnv(
            EnvConfig(
                env_name=env_name,
                n_objects_min=n_min,
                n_objects_max=n_max,
                candidate_max_dist=2,
                seed=seed,
            )
        )
        self.records: List[Dict[str, Any]] = []
        self.meta_rows: List[Dict[str, Any]] = []

    def collect(self) -> None:
        n_episodes = self.samples // RECORDS_PER_EPISODE
        for episode_id in range(n_episodes):
            self.env.reset()
            self.env.episode_idx = episode_id
            self._make_episode_ambiguous()
            start = time.time()
            meta = {
                "episode_id": episode_id,
                "env_name": self.env_name,
                "wizard_id": WIZARD_ID,
                "intended_obj_id": self.env.intended_obj_id,
                "started_at": start,
                "decisions": [],
            }
            if self.env_name == "reach_to_grasp_ycb":
                self._collect_ycb_episode(meta)
            elif self.env_name == "cube_stacking":
                self._collect_stacking_episode(meta)
            elif self.env_name == "pouring":
                self._collect_pouring_episode(meta)
            else:
                raise ValueError(f"Unsupported env: {self.env_name}")
            meta["ended_at"] = time.time()
            meta["n_decisions"] = len(meta["decisions"])
            self.meta_rows.append(meta)

    def write(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(self.output_dir / "grasp_gen.jsonl", self.records)
        _write_jsonl(self.output_dir / "episodes_meta.jsonl", self.meta_rows)
        summary = self._summary()
        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    def _summary(self) -> Dict[str, Any]:
        tools = Counter(r["target_tool_call"]["tool"] for r in self.records)
        interact_kinds = Counter(
            r["target_tool_call"]["args"]["kind"]
            for r in self.records
            if r["target_tool_call"]["tool"] == "INTERACT"
        )
        replies = Counter()
        for record in self.records:
            for dialog in record.get("memory", {}).get("past_dialogs", []):
                reply = str(dialog.get("reply") or "").upper()
                if "YES" in reply:
                    replies["YES"] += 1
                if "NO" in reply:
                    replies["NO"] += 1
                if "NONE" in reply:
                    replies["NONE_OF_THEM"] += 1
                if "OK" in reply:
                    replies["OK"] += 1
        return {
            "env_name": self.env_name,
            "wizard_id": WIZARD_ID,
            "seed": self.seed,
            "records": len(self.records),
            "episodes": len(self.meta_rows),
            "records_per_episode": RECORDS_PER_EPISODE,
            "tool_distribution": dict(sorted(tools.items())),
            "interact_kind_distribution": dict(sorted(interact_kinds.items())),
            "memory_reply_distribution": dict(sorted(replies.items())),
            "output_dir": str(self.output_dir),
        }

    def _snapshot(self) -> Dict[str, Any]:
        blob = self.env.public_blob()
        memory = blob["memory"]
        memory["candidates"] = list(self.env.memory.get("candidates") or [])
        memory["excluded_obj_ids"] = list(self.env.memory.get("excluded_obj_ids") or [])
        return blob

    def _record(self, tool_call: Dict[str, Any], meta: Dict[str, Any], alert_reason: str) -> None:
        validate_tool_call(tool_call, env=self.env_name)
        blob = self._snapshot()
        self.records.append(
            {
                "episode_id": meta["episode_id"],
                "objects": blob["objects"],
                "gripper_hist": blob["gripper_hist"],
                "memory": blob["memory"],
                "user_state": blob["user_state"],
                "target_tool_call": tool_call,
            }
        )
        meta["decisions"].append(
            {
                "tick_idx": len(meta["decisions"]),
                "alert_reason": alert_reason,
                "tool_call": tool_call,
            }
        )

    def _interact(
        self,
        prompt_type: str,
        meta: Dict[str, Any],
        *,
        reply_target_id: Optional[str] = None,
        alert_reason: str = "self_woz_interact",
        forced_reply: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], str]:
        blob = self._snapshot()
        tool_call, context = build_prompt(self.env_name, prompt_type, blob, **kwargs)
        self._record(tool_call, meta, alert_reason)
        reply = self._semantic_reply(
            tool_call,
            context,
            reply_target_id=reply_target_id,
            forced_reply=forced_reply,
        )
        self._apply_interaction(tool_call, context, reply, reply_target_id=reply_target_id)
        return tool_call, reply

    def _execute(
        self,
        tool: str,
        meta: Dict[str, Any],
        *,
        obj_id: str = "",
        amount: Optional[str] = None,
        alert_reason: str = "self_woz_execute",
    ) -> None:
        args: Dict[str, Any] = {}
        if obj_id:
            args["obj"] = obj_id
        if amount is not None:
            args["amount"] = amount
        tool_call = {"tool": tool, "args": args}
        self._record(tool_call, meta, alert_reason)
        self.env.apply_execution_skill(tool, obj_id, amount)

    def _semantic_reply(
        self,
        tool_call: Dict[str, Any],
        context: Dict[str, Any],
        *,
        reply_target_id: Optional[str],
        forced_reply: Optional[str] = None,
    ) -> str:
        args = tool_call["args"]
        choices = list(args.get("choices") or [])
        if not choices:
            return "1) OK"
        if forced_reply:
            forced = forced_reply.strip().upper()
            semantics = [c.split(")", 1)[-1].strip().upper() for c in choices]
            for idx, semantic in enumerate(semantics):
                if semantic == forced or forced in semantic:
                    return choices[idx]
            for c in choices:
                if forced in c.upper():
                    return c
            return choices[0]
        target_label = ""
        if reply_target_id:
            try:
                target_label = _obj_by_id(self._snapshot(), reply_target_id).get("label", "")
            except KeyError:
                target_label = ""
        ctx_type = str(context.get("type") or "")

        if ctx_type in {"candidate_choice", "amount_choice", "mode_select"}:
            if ctx_type == "amount_choice":
                preferred = self.rng.choice(["HALF", "FULL", "SMALL"])
                return next((c for c in choices if preferred in c), choices[0])
            if ctx_type == "mode_select":
                priority = ("STACK", "POUR", "GRAB", "APPROACH", "ALIGN_YAW")
                for action in priority:
                    for c in choices:
                        if action in c:
                            return c
            if target_label:
                return next((c for c in choices if target_label.lower() in c.lower()), choices[-1])

        semantics = [c.split(")", 1)[-1].strip().upper() for c in choices]
        if ctx_type == "anything_else" and "NO" in semantics:
            return choices[semantics.index("NO")]
        if "YES" in semantics:
            return choices[semantics.index("YES")]
        if "OK" in semantics:
            return choices[semantics.index("OK")]
        if "NO" in semantics:
            return choices[semantics.index("NO")]
        return choices[0]

    def _apply_interaction(
        self,
        tool_call: Dict[str, Any],
        context: Dict[str, Any],
        reply: str,
        *,
        reply_target_id: Optional[str],
    ) -> None:
        args = tool_call["args"]
        kind = args.get("kind", "QUESTION")
        text = args.get("text", "")
        choices = list(args.get("choices") or [])
        self.env.apply_interaction(kind, text, choices, reply)
        self.env.memory["last_prompt"] = {
            "kind": kind,
            "text": text,
            "choices": choices,
            "context": dict(context),
        }

        ctx_type = str(context.get("type") or "")
        if ctx_type == "candidate_choice" and reply_target_id:
            self._select_single_candidate(reply_target_id)
        elif ctx_type in {"anything_else", "terminal_ack"}:
            self.env.memory["candidates"] = list(self.env.memory.get("candidates") or [])
        elif "NO" in reply.upper() and reply_target_id:
            excluded = set(self.env.memory.get("excluded_obj_ids") or [])
            excluded.add(reply_target_id)
            self.env.memory["excluded_obj_ids"] = sorted(excluded)
            self.env.memory["candidates"] = [
                c for c in self.env.memory.get("candidates", []) if c != reply_target_id
            ]

    def _select_single_candidate(self, target_id: str) -> None:
        excluded = set(self.env.memory.get("excluded_obj_ids") or [])
        for obj in self.env.objects:
            if not obj.is_held and obj.id != target_id:
                excluded.add(obj.id)
        self.env.memory["excluded_obj_ids"] = sorted(excluded)
        self.env.memory["candidates"] = [target_id]

    def _make_episode_ambiguous(self) -> None:
        target = next(o for o in self.env.objects if o.id == self.env.intended_obj_id)
        self.env.gripper.cell = target.cell
        self.env.gripper.yaw = self.rng.choice(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
        self.env.gripper.z = "MID"
        self.env.gripper_hist.clear()
        for _ in range(self.env.cfg.history_len):
            self.env.gripper_hist.append(self.env._gripper_record())

        candidates = [target.id]
        for obj in self.env.objects:
            if obj.id == target.id or obj.is_held:
                continue
            if self.env_name == "pouring" and obj.kind != "cup":
                continue
            obj.cell = target.cell
            candidates.append(obj.id)
            if len(candidates) >= 3:
                break
        self.env.memory["excluded_obj_ids"] = []
        self.env.memory["candidates"] = candidates

    def _alternate_candidate_id(self, target_id: str, *, require_kind: Optional[str] = None) -> str:
        for obj in self.env.objects:
            if obj.id == target_id or obj.is_held:
                continue
            if require_kind is not None and obj.kind != require_kind:
                continue
            return obj.id
        return target_id

    def _collect_ycb_episode(self, meta: Dict[str, Any]) -> None:
        target_id = self.env.intended_obj_id
        variant = meta["episode_id"] % 4
        if variant == 1:
            wrong_id = self._alternate_candidate_id(target_id)
            self._interact(
                "confirm",
                meta,
                reply_target_id=wrong_id,
                target_id=wrong_id,
                action="APPROACH",
                forced_reply="NO",
                alert_reason="negative_wrong_target",
            )
            self._interact("candidate_choice", meta, reply_target_id=target_id, alert_reason="recover_prune_candidates")
            self._interact(
                "confirm",
                meta,
                reply_target_id=target_id,
                target_id=target_id,
                action="APPROACH",
                alert_reason="confirm_corrected_action",
            )
            self._execute("APPROACH", meta, obj_id=target_id)
            self._interact("help", meta, reply_target_id=target_id, target_id=target_id, alert_reason="followup_suggestion")
            return
        if variant == 2:
            self._interact("candidate_choice", meta, forced_reply="None of them", alert_reason="negative_none_of_them")
            self._interact("anything_else", meta, forced_reply="YES", alert_reason="recover_anything_else")
            self._interact("candidate_choice", meta, reply_target_id=target_id, alert_reason="recover_prune_candidates")
            self._interact(
                "confirm",
                meta,
                reply_target_id=target_id,
                target_id=target_id,
                action="APPROACH",
                alert_reason="confirm_corrected_action",
            )
            self._execute("APPROACH", meta, obj_id=target_id)
            return
        self._interact("candidate_choice", meta, reply_target_id=target_id, alert_reason="prune_candidates")
        self._interact(
            "confirm",
            meta,
            reply_target_id=target_id,
            target_id=target_id,
            action="APPROACH",
            alert_reason="confirm_before_action",
        )
        self._execute("APPROACH", meta, obj_id=target_id)
        self._interact("help", meta, reply_target_id=target_id, target_id=target_id, alert_reason="followup_suggestion")
        self._execute("ALIGN_YAW", meta, obj_id=target_id)

    def _collect_stacking_episode(self, meta: Dict[str, Any]) -> None:
        target_id = self.env.intended_obj_id
        variant = meta["episode_id"] % 4
        if variant == 1:
            wrong_id = self._alternate_candidate_id(target_id)
            self._interact(
                "confirm_stack",
                meta,
                reply_target_id=wrong_id,
                target_id=wrong_id,
                forced_reply="NO",
                alert_reason="negative_wrong_target",
            )
            self._interact("candidate_choice", meta, reply_target_id=target_id, alert_reason="recover_prune_candidates")
            self._interact(
                "confirm_stack",
                meta,
                reply_target_id=target_id,
                target_id=target_id,
                alert_reason="confirm_corrected_action",
            )
            self._execute("STACK", meta, obj_id=target_id)
            self._select_single_candidate(target_id)
            self._interact("anything_else", meta, alert_reason="followup_suggestion")
            return
        if variant == 2:
            self._interact("candidate_choice", meta, forced_reply="None of them", alert_reason="negative_none_of_them")
            self._interact("anything_else", meta, forced_reply="YES", alert_reason="recover_anything_else")
            self._interact("candidate_choice", meta, reply_target_id=target_id, alert_reason="recover_prune_candidates")
            self._interact(
                "confirm_stack",
                meta,
                reply_target_id=target_id,
                target_id=target_id,
                alert_reason="confirm_corrected_action",
            )
            self._execute("STACK", meta, obj_id=target_id)
            return
        self._interact("candidate_choice", meta, reply_target_id=target_id, alert_reason="prune_candidates")
        self._interact(
            "confirm_stack",
            meta,
            reply_target_id=target_id,
            target_id=target_id,
            alert_reason="confirm_before_action",
        )
        self._execute("STACK", meta, obj_id=target_id)
        self._select_single_candidate(target_id)
        self._interact("anything_else", meta, alert_reason="followup_suggestion")
        self._interact("terminal_ack", meta, alert_reason="close_conversation")

    def _collect_pouring_episode(self, meta: Dict[str, Any]) -> None:
        cup_id = self.env.intended_obj_id
        pitcher = next(o for o in self.env.objects if o.kind == "pitcher")
        variant = meta["episode_id"] % 4
        if variant == 1:
            self._interact("candidate_choice", meta, reply_target_id=cup_id, alert_reason="prune_candidates")
            self._interact(
                "pitcher_acquisition",
                meta,
                reply_target_id=pitcher.id,
                target_id=pitcher.id,
                forced_reply="NO",
                alert_reason="negative_grab_declined",
            )
            self._interact("anything_else", meta, forced_reply="YES", alert_reason="recover_anything_else")
            self._interact(
                "pitcher_acquisition",
                meta,
                reply_target_id=pitcher.id,
                target_id=pitcher.id,
                alert_reason="confirm_corrected_action",
            )
            self._execute("GRAB", meta, obj_id=pitcher.id)
            return
        if variant == 2:
            self._interact("candidate_choice", meta, reply_target_id=cup_id, alert_reason="prune_candidates")
            self._interact(
                "pitcher_acquisition",
                meta,
                reply_target_id=pitcher.id,
                target_id=pitcher.id,
                alert_reason="confirm_before_action",
            )
            self._execute("GRAB", meta, obj_id=pitcher.id)
            self._select_single_candidate(cup_id)
            amount = self.rng.choice(["SMALL", "HALF", "FULL"])
            self._interact(
                "confirm_amount",
                meta,
                reply_target_id=cup_id,
                target_id=cup_id,
                amount=amount,
                forced_reply="NO",
                alert_reason="negative_amount_declined",
            )
            self._interact("amount_choice", meta, reply_target_id=cup_id, target_id=cup_id, alert_reason="recover_amount_choice")
            return
        self._interact("candidate_choice", meta, reply_target_id=cup_id, alert_reason="prune_candidates")
        self._interact(
            "pitcher_acquisition",
            meta,
            reply_target_id=pitcher.id,
            target_id=pitcher.id,
            alert_reason="confirm_before_action",
        )
        self._execute("GRAB", meta, obj_id=pitcher.id)
        self._select_single_candidate(cup_id)
        amount = self.rng.choice(["SMALL", "HALF", "FULL"])
        self._interact(
            "confirm_amount",
            meta,
            reply_target_id=cup_id,
            target_id=cup_id,
            amount=amount,
            alert_reason="followup_confirm_before_action",
        )
        self._execute("POUR", meta, obj_id=cup_id, amount=amount)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--samples-per-env", type=int, default=DEFAULT_SAMPLES_PER_ENV)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--env", choices=[*ENVS, "all"], default="all")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    envs = ENVS if args.env == "all" else (args.env,)
    summaries = []
    for idx, env_name in enumerate(envs):
        collector = SelfWozCollector(
            env_name,
            samples=int(args.samples_per_env),
            seed=int(args.seed) + idx * 1009,
            output_root=args.output_root,
        )
        collector.collect()
        collector.write()
        summaries.append(collector._summary())

    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
