"""
Episode-level rollout driver — the inner loop of the noise sweep.

Per plan §5.6, one rollout = (scenario × mode × noise condition × seed).

Modes
-----
  * ``manual``  — no LLM. The scripted user drives the gripper end-to-end
                  using translation / rotation / gripper toggles. No tool
                  calls, no INTERACT prompts. Selection noise is unused.
  * ``prime``   — PRIME (an LLM or oracle) decides at every tick whether to
                  ACT (APPROACH/ALIGN_YAW) or ASK (INTERACT). The scripted
                  user answers prompts; on motion ticks the user emits its
                  own command for translation/rotation. The backend is
                  supplied by the caller; the simplest one is the heuristic
                  oracle from `data_generator/oracle.py`.

The backend protocol is a callable
    decide(input_dict: dict) -> dict | None
returning a tool-call dict ``{"tool": ..., "args": ...}`` or None to defer
to the scripted user's motion command.

This module is deliberately backend-agnostic — wiring up the fine-tuned
Qwen model lives in the sweep runner (§5.7), not here.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from data_generator.episode import Episode
from evaluation.rollouts.noise import NoiseInjector, NoiseProfile
from evaluation.rollouts.scripted_user import ScriptedUser


# ── memory book-keeping (mirrors data_generator/episode.py + adapters) ──────


def _initial_memory(episode: Episode, candidate_max_dist: int = 1) -> Dict[str, Any]:
    cur = episode.gripper_hist[-1]
    cands: List[str] = []
    from data_generator import grid as gridlib
    for o in episode.objects:
        if o.is_held:
            continue
        if gridlib.manhattan(cur.cell, o.cell) <= candidate_max_dist:
            cands.append(o.id)
    return {
        "n_interactions": 0,
        "past_dialogs": [],
        "candidates": cands,
        "last_tool_calls": [],
        "excluded_obj_ids": [],
        "last_action": {},
    }


def _refresh_candidates(memory: Dict[str, Any], episode: Episode, candidate_max_dist: int = 1) -> None:
    from data_generator import grid as gridlib
    cur = episode.gripper_hist[-1]
    memory["candidates"] = [
        o.id for o in episode.objects
        if not o.is_held and gridlib.manhattan(cur.cell, o.cell) <= candidate_max_dist
    ]


def _build_llm_input(episode: Episode, memory: Dict[str, Any], user_mode: str) -> Dict[str, Any]:
    return {
        "objects": [o.to_record() for o in episode.objects],
        "gripper_hist": [p.to_record() for p in episode.gripper_hist],
        "memory": memory,
        "user_state": {"mode": user_mode},
    }


# ── result record ───────────────────────────────────────────────────────────


@dataclass
class RolloutResult:
    scenario_id: str
    mode: str                    # "manual" | "prime"
    condition: str               # NoiseProfile.name
    seed: int

    success: bool                = False
    completion_time_sec: float   = 0.0
    total_inputs: int            = 0
    interactions: int            = 0
    motion_tool_calls: int       = 0
    mode_switches: int           = 0
    direction_reversals: int     = 0
    dropped_inputs: int          = 0
    selection_perturbations: int = 0
    direction_perturbations: int = 0
    target_filtered_out: bool    = False   # PRIME ever omitted target from a prompt
    terminated_at_max_ticks: bool = False
    end_reason: str              = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── main entry point ────────────────────────────────────────────────────────


def run_rollout(
    scenario,
    mode: str,
    noise_profile: NoiseProfile,
    *,
    seed: int,
    backend: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
    max_ticks: int = 2000,
    candidate_max_dist: int = 1,
    user_mode_default: str = "translation",
    hesitation_rate: float = 0.05,
    mode_switch_cost_sec: float = 0.6,
    tick_dt_sec: float = 0.3,
    use_per_scenario_priors: bool = True,
) -> RolloutResult:
    """Run a single (scenario, mode, noise) episode rollout.

    Args:
        scenario: a Scenario record (dataclass instance OR dict).
        mode: "manual" or "prime".
        noise_profile: the NoiseProfile to inject through.
        seed: rollout seed (drives sim RNG, user RNG, and noise RNG).
        backend: required only for mode="prime". Callable
            `(input_dict) -> tool_call_dict | None`. None means "defer to
            user's motion command for this tick".
        max_ticks: hard cap on simulator ticks; rollout terminates as failure
            if exceeded. Defaults to 2000 to accommodate priors-calibrated
            users who emit ~10 bursts per logical step.
        candidate_max_dist: Manhattan radius for the candidate set
            (mirrors `Episode.gripper_candidates`).
        user_mode_default: starting user_state.mode value.
        hesitation_rate, mode_switch_cost_sec, tick_dt_sec: fallback knobs used
            only when `use_per_scenario_priors=False` (or priors are absent).
        use_per_scenario_priors: if True (default), build the scripted user
            from `scenario.user_priors` so its emit rate, mean burst time, and
            reversal rate match the real user who ran that trial. See
            `ScriptedUser.from_priors`.

    Returns:
        RolloutResult with per-trial metrics.
    """
    if mode not in ("manual", "prime"):
        raise ValueError(f"mode must be 'manual' or 'prime' (got {mode!r})")
    if mode == "prime" and backend is None:
        raise ValueError("mode='prime' requires a backend callable")

    scenario_id = getattr(scenario, "scenario_id", None) or scenario.get("scenario_id", "?")
    target_id = getattr(scenario, "target_obj_id", None) or scenario.get("target_obj_id")
    if target_id is None:
        raise ValueError(f"scenario {scenario_id} has no target_obj_id (label first)")

    sim = Episode.from_scenario(scenario, rng=random.Random(seed))

    priors = None
    if use_per_scenario_priors:
        priors = (
            getattr(scenario, "user_priors", None)
            or (scenario.get("user_priors") if isinstance(scenario, dict) else None)
        )

    if priors is not None:
        user = ScriptedUser.from_priors(
            target_obj_id=target_id,
            rng=random.Random(seed ^ 0xA5A5A5A5),
            priors=priors,
            scenario=scenario,
            mode_switch_cost_sec=mode_switch_cost_sec,
        )
    else:
        user = ScriptedUser(
            target_obj_id=target_id,
            rng=random.Random(seed ^ 0xA5A5A5A5),
            hesitation_rate=hesitation_rate,
            mode_switch_cost_sec=mode_switch_cost_sec,
            tick_dt_sec=tick_dt_sec,
        )
    noise = NoiseInjector(profile=noise_profile, rng=random.Random(seed ^ 0x5A5A5A5A))
    memory = _initial_memory(sim, candidate_max_dist=candidate_max_dist)

    result = RolloutResult(
        scenario_id=scenario_id,
        mode=mode,
        condition=noise_profile.name,
        seed=seed,
    )

    last_axis_dir: Dict[tuple, int] = {}
    current_user_mode = user_mode_default

    for tick in range(max_ticks):
        if user.is_done(sim):
            result.success = True
            result.end_reason = "task_complete"
            break

        # PRIME backend gets a chance to act/ask before the user moves.
        if mode == "prime":
            llm_input = _build_llm_input(sim, memory, current_user_mode)
            tool_call = backend(llm_input)
        else:
            tool_call = None

        if tool_call is not None:
            tool = tool_call.get("tool")
            if tool == "INTERACT":
                args = tool_call.get("args", {}) or {}
                choices = list(args.get("choices") or [])
                prompt_text = args.get("text", "")
                kind = args.get("kind", "QUESTION")

                # Scripted user answers; selection noise may flip the answer.
                reply_idx, target_in_options = user.answer_prompt(tool_call, sim)
                reply_idx = noise.selection(reply_idx, len(choices))
                reply_str = choices[reply_idx] if 0 <= reply_idx < len(choices) else ""

                # Format must match the training-time contract (LLM was trained
                # on {prompt, kind, choices, reply}, NOT reply_idx).
                memory["past_dialogs"].append({
                    "prompt":  prompt_text,
                    "kind":    kind,
                    "choices": choices,
                    "reply":   reply_str,
                })
                memory["n_interactions"] += 1
                memory["last_tool_calls"] = (memory["last_tool_calls"] + ["INTERACT"])[-3:]

                # Record last_prompt so the next decide() can see what was just
                # asked. For stateful backends (oracle), pull the structured
                # context from the backend's internal state. For stateless
                # backends (LLM), context is unavailable — the LLM has to
                # infer it from kind/text/choices alone (training data has
                # context, deployment may be more tolerant).
                last_prompt: Dict[str, Any] = {"kind": kind, "text": prompt_text, "choices": choices}
                backend_state = getattr(backend, "state", None)
                if backend_state is not None and getattr(backend_state, "last_prompt_context", None):
                    last_prompt["context"] = dict(backend_state.last_prompt_context)
                memory["last_prompt"] = last_prompt

                result.interactions += 1
                if not target_in_options:
                    result.target_filtered_out = True

                # Stateful backends need the reply fed back in to update their
                # internal awaiting_* flags / candidate set. Stateless backends
                # (LLM) don't need this — they re-read memory.past_dialogs each
                # call.
                on_reply = getattr(backend, "on_user_reply", None)
                if on_reply is not None:
                    try:
                        on_reply(tool_call, reply_idx, memory, sim)
                    except Exception:
                        pass
                result.completion_time_sec += user.tick_dt_sec  # cost of an exchange
                continue

            if tool in ("APPROACH", "ALIGN_YAW"):
                try:
                    sim.apply_tool(tool_call)
                except Exception:
                    pass  # treat malformed tool calls as no-ops
                result.motion_tool_calls += 1
                memory["last_tool_calls"] = (memory["last_tool_calls"] + [tool])[-3:]
                memory["last_action"] = {
                    "tool": tool,
                    "obj": tool_call.get("args", {}).get("obj"),
                    "outcome": "success",
                }
                _refresh_candidates(memory, sim, candidate_max_dist=candidate_max_dist)
                # Motion tool execution is "instant" in the sim but takes a
                # realistic chunk of wall-clock time.
                result.completion_time_sec += 1.5
                continue

            # Unknown tool — fall through to user command.

        # User motion command.
        cmd = user.next_command(sim)
        if cmd is None:
            # Shouldn't happen unless user.is_done() races with state changes.
            result.end_reason = "user_no_command"
            break

        # Track direction reversals BEFORE noise so the metric reflects user intent.
        if cmd.get("mode") in ("translation", "rotation"):
            key = (cmd["mode"], cmd.get("axis"))
            prev = last_axis_dir.get(key)
            if prev is not None and prev != cmd.get("direction"):
                result.direction_reversals += 1
            if cmd.get("direction") != 0:
                last_axis_dir[key] = cmd["direction"]
        if current_user_mode != cmd["mode"]:
            result.mode_switches += 1
            current_user_mode = cmd["mode"]

        # Apply noise: direction perturb, then dropout.
        cmd_perturbed = noise.direction(cmd)
        cmd_after = noise.dropout(cmd_perturbed)

        if cmd_after is None:
            result.dropped_inputs += 1
            # Missed input still costs a tick of wall-clock.
            result.completion_time_sec += user.tick_dt_sec + noise.latency_sec()
            continue

        # Decide whether to mutate simulator state. Tracking bursts and
        # hesitation-flipped bursts carry `_advance_sim=False` — they cost
        # input count + time but don't move the gripper. Direction noise can
        # only meaningfully fire on the terminal burst, since intermediate
        # bursts wouldn't advance state anyway.
        should_advance = bool(cmd_after.get("_advance_sim", True))
        if should_advance:
            try:
                sim.step_user_command(
                    axis=cmd_after.get("axis", ""),
                    direction=int(cmd_after.get("direction", 0)),
                    mode=cmd_after["mode"],
                )
            except Exception:
                result.dropped_inputs += 1
                result.completion_time_sec += user.tick_dt_sec
                continue
            _refresh_candidates(memory, sim, candidate_max_dist=candidate_max_dist)

        result.total_inputs += 1
        result.completion_time_sec += user.tick_cost_sec(cmd) + noise.latency_sec()
    else:
        # while-else: ran out of max_ticks without breaking.
        result.terminated_at_max_ticks = True
        result.end_reason = "max_ticks"

    if user.is_done(sim) and not result.success:
        result.success = True
        result.end_reason = "task_complete"

    # Pull noise-channel stats from the injector.
    s = noise.stats()
    result.direction_perturbations = s["direction_perturbed"]
    result.selection_perturbations = s["selection_perturbed"]
    return result


__all__ = ["RolloutResult", "run_rollout"]
