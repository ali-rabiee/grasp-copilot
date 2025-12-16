import json
from pathlib import Path

import generate_dataset
from oracle import validate_tool_call


def test_generated_jsonl_schema(tmp_path: Path):
    out = tmp_path / "sample.jsonl"
    records, stats = generate_dataset.generate(episodes=2, seed=0)
    out.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    assert "tool_distribution" in stats
    assert "avg_episode_length" in stats
    assert "ambiguity_rate" in stats
    assert "grasp_success_rate" in stats
    assert "fraction_just_guide_episodes" in stats

    with out.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            assert set(rec.keys()) == {"episode_id", "t", "obs", "dialog", "target_tool_call"}
            obs = rec["obs"]
            assert len(obs["gripper_hist"]) == 6
            validate_tool_call(rec["target_tool_call"])
            candidates = set(obs["candidates"])
            not_held = {o["id"] for o in obs["objects"] if not o["is_held"]}
            assert candidates == not_held

