import json
from pathlib import Path

from data_generator import generate_dataset
from data_generator import grid
from data_generator.oracle import validate_tool_call


def test_generated_jsonl_schema(tmp_path: Path):
    out = tmp_path / "sample.jsonl"
    records, stats = generate_dataset.generate(episodes=2, seed=0)
    out.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    assert "tool_distribution" in stats

    with out.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            assert set(rec.keys()) == {"episode_id", "objects", "gripper_hist", "memory", "user_state", "target_tool_call"}
            assert len(rec["gripper_hist"]) == 6
            validate_tool_call(rec["target_tool_call"])
            assert rec["user_state"]["mode"] in {"translation", "rotation", "gripper"}

            grip_cell = rec["gripper_hist"][-1]["cell"]
            for cid in rec["memory"]["candidates"]:
                obj = next(o for o in rec["objects"] if o["id"] == cid)
                assert grid.manhattan(grip_cell, obj["cell"]) <= 1
