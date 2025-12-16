from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

from episode import Episode, OBJECT_LABELS, write_jsonl
from oracle import OracleState, oracle_decide_tool, validate_tool_call


def _append_user_response_for_ambiguity(
    rng: random.Random,
    dialog: List[Dict],
    choices: Tuple[str, str],
    intended_label: str,
) -> None:
    if rng.random() >= 0.6:
        return
    a, b = choices
    if intended_label in choices and rng.random() < 0.7:
        pick = intended_label
    else:
        pick = rng.choice([a, b])
    dialog.append({"role": "user", "content": pick})


def _append_user_response_for_takeover(rng: random.Random, dialog: List[Dict]) -> bool:
    """
    Returns True if user says "no, just guide", else False.
    """
    if rng.random() < 0.5:
        dialog.append({"role": "user", "content": "yes please"})
        return False
    dialog.append({"role": "user", "content": "no, just guide"})
    return True


def _schema_validate_record(rec: Dict) -> None:
    for k in ("episode_id", "t", "obs", "dialog", "target_tool_call"):
        if k not in rec:
            raise ValueError(f"Missing key: {k}")
    obs = rec["obs"]
    if set(obs.keys()) != {"objects", "gripper_hist", "candidates", "last_action_outcome"}:
        raise ValueError("obs keys mismatch")
    if len(obs["gripper_hist"]) != 6:
        raise ValueError("gripper_hist must have length 6")
    validate_tool_call(rec["target_tool_call"])


def generate(episodes: int, seed: int) -> Tuple[List[Dict], Dict]:
    rng = random.Random(seed)
    records: List[Dict] = []

    tool_counts: Counter = Counter()
    total_steps = 0
    ambiguity_steps = 0
    grasp_attempts = 0
    grasp_successes = 0
    just_guide_episodes = 0

    for episode_id in range(episodes):
        n_obj = rng.randint(2, len(OBJECT_LABELS))
        ep = Episode(rng=rng, episode_id=episode_id, n_obj=n_obj)
        intended_label = ep.intended_obj().label

        dialog: List[Dict] = []
        state = OracleState()
        last_tool_name: Optional[str] = None
        episode_just_guide = False

        for t in range(ep.T):
            # Simulated user replies arrive one step after the assistant prompt.
            if state.pending_user_prompt == "ambiguity" and state.pending_ambiguity_choices is not None:
                _append_user_response_for_ambiguity(rng, dialog, state.pending_ambiguity_choices, intended_label)
                state.pending_user_prompt = None
                state.pending_ambiguity_choices = None
            elif state.pending_user_prompt == "takeover":
                said_no = _append_user_response_for_takeover(rng, dialog)
                state.pending_user_prompt = None
                state.pending_takeover_obj_id = None
                if said_no:
                    state.just_guide = True
                    episode_just_guide = True

            # User teleop steps happen when the assistant isn't directly controlling motion.
            if last_tool_name in (None, "INTERACT", "SELECT_TARGET", "RETRY_OR_ABORT"):
                ep.apply_user_teleop_step()

            obs = ep.observe()
            tool_call = oracle_decide_tool(obs=obs, dialog=dialog, state=state)
            validate_tool_call(tool_call)

            rec = {
                "episode_id": episode_id,
                "t": t,
                "obs": obs,
                "dialog": list(dialog),
                "target_tool_call": tool_call,
            }
            _schema_validate_record(rec)
            records.append(rec)

            tool_counts[tool_call["tool_name"]] += 1
            total_steps += 1
            if tool_call["tool_name"] == "INTERACT" and tool_call["arguments"]["type"] == "question":
                ambiguity_steps += 1
                state.pending_user_prompt = "ambiguity"
                choices = tool_call["arguments"].get("choices", [])
                if isinstance(choices, list) and len(choices) == 2:
                    state.pending_ambiguity_choices = (choices[0], choices[1])
            if tool_call["tool_name"] == "INTERACT" and tool_call["arguments"]["type"] == "offer_takeover":
                state.pending_user_prompt = "takeover"

            if tool_call["tool_name"] == "GRASP":
                grasp_attempts += 1

            outcome = ep.step_tool(tool_call)
            state.outcomes.append(outcome)
            last_tool_name = tool_call["tool_name"]

            if outcome == "grasp_success":
                grasp_successes += 1

            # Update dialog after the assistant action for next timestep context.
            if tool_call["tool_name"] == "INTERACT":
                dialog.append({"role": "assistant", "content": tool_call["arguments"]["text"]})

        if episode_just_guide:
            just_guide_episodes += 1

    stats = {
        "tool_distribution": dict(tool_counts),
        "avg_episode_length": (total_steps / episodes) if episodes else 0.0,
        "ambiguity_rate": (ambiguity_steps / total_steps) if total_steps else 0.0,
        "grasp_success_rate": (grasp_successes / grasp_attempts) if grasp_attempts else 0.0,
        "fraction_just_guide_episodes": (just_guide_episodes / episodes) if episodes else 0.0,
    }
    return records, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    records, stats = generate(episodes=args.episodes, seed=args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_jsonl(args.out, records)
    with open(args.out + ".stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
