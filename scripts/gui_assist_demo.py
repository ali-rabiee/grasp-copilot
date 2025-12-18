"""
Minimal GUI playground to visually validate the grasp-copilot policy.

- White canvas shows a 3x3 grid (A1..C3), object locations + labels, and the gripper pose.
- Use keyboard to move the gripper:
  - Arrow keys: move cell (4-connected)
  - Q/E: rotate yaw (counter-clockwise / clockwise)
  - W/S: change z (HIGH/MID/LOW)
- Click "Ask assistance" to query either:
  - the oracle policy (fast debug), or
  - a HuggingFace model via llm.inference (your trained adapter/merged model).

Run:
  conda activate talm
  python scripts/gui_assist_demo.py --backend oracle

  # HF backend (example; update paths as needed):
  python scripts/gui_assist_demo.py --backend hf --model_name Qwen/Qwen2.5-7B-Instruct --adapter_path outputs/qwen_lora
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import _bootstrap  # noqa: F401

from data_generator import grid as gridlib
from data_generator import yaw as yawlib
from data_generator.episode import OBJECT_LABELS, Z_BINS
from data_generator.oracle import OracleState, oracle_decide_tool, validate_tool_call
from llm.inference import InferenceConfig, generate_json_only


INSTRUCTION = (
    "Given the robot observation and dialog context, infer the user's intent and "
    "emit exactly one tool call. Output ONLY the tool call JSON with keys tool and args."
)


def _strip_choice_label(choice: str) -> str:
    parts = choice.split(")", 1)
    return parts[1].strip() if len(parts) == 2 else choice.strip()


def _yaw_dir(yaw: str) -> Tuple[float, float]:
    # Canvas y increases downward; treat N as up.
    yaw = yawlib.normalize(yaw)
    dirs: Dict[str, Tuple[float, float]] = {
        "N": (0.0, -1.0),
        "NE": (0.7, -0.7),
        "E": (1.0, 0.0),
        "SE": (0.7, 0.7),
        "S": (0.0, 1.0),
        "SW": (-0.7, 0.7),
        "W": (-1.0, 0.0),
        "NW": (-0.7, -0.7),
    }
    return dirs[yaw]


@dataclass
class Obj:
    id: str
    label: str
    cell: str
    yaw: str
    is_held: bool = False

    def to_record(self) -> Dict[str, Any]:
        return {"id": self.id, "label": self.label, "cell": self.cell, "yaw": self.yaw, "is_held": self.is_held}


@dataclass
class Pose:
    cell: str
    yaw: str
    z: str

    def to_record(self) -> Dict[str, Any]:
        return {"cell": self.cell, "yaw": self.yaw, "z": self.z}


class GridWorld:
    def __init__(self, rng: random.Random, *, n_obj: int, collision_p: float) -> None:
        labels = list(OBJECT_LABELS)
        rng.shuffle(labels)
        labels = labels[: max(2, min(n_obj, len(labels)))]

        objs: List[Obj] = []
        for i, label in enumerate(labels):
            if objs and rng.random() < collision_p:
                cell = rng.choice([o.cell for o in objs])
            else:
                cell = rng.choice(list(gridlib.CELLS))
            yaw = rng.choice(list(yawlib.YAW_BINS))
            objs.append(Obj(id=f"o{i}", label=label, cell=cell, yaw=yaw))
        self.objects = objs

        init_pose = Pose(
            cell=rng.choice(list(gridlib.CELLS)),
            yaw=rng.choice(list(yawlib.YAW_BINS)),
            z=rng.choice(list(Z_BINS)),
        )
        # Option B: warm-start history with motion toward a random "intended" object to mimic exploration.
        intended = rng.choice(self.objects).id
        self.intended_obj_id = intended
        self.gripper_hist: List[Pose] = [init_pose]
        while len(self.gripper_hist) < 6:
            self._apply_user_motion_toward(intended_obj_id=intended, rng=rng)

    def get_obj(self, obj_id: str) -> Obj:
        for o in self.objects:
            if o.id == obj_id:
                return o
        raise KeyError(obj_id)

    def candidates(self, max_dist: int) -> List[str]:
        cell = self.gripper_hist[-1].cell
        out: List[str] = []
        for o in self.objects:
            if o.is_held:
                continue
            if gridlib.manhattan(cell, o.cell) <= max_dist:
                out.append(o.id)
        return out

    def push_gripper(self, pose: Pose) -> None:
        self.gripper_hist.append(pose)
        self.gripper_hist = self.gripper_hist[-6:]

    def apply_tool(self, tool_call: Dict[str, Any]) -> None:
        tool = tool_call["tool"]
        args = tool_call["args"]
        cur = self.gripper_hist[-1]
        if tool == "INTERACT":
            return
        if tool == "APPROACH":
            obj = self.get_obj(args["obj"])
            self.push_gripper(Pose(cell=obj.cell, yaw=cur.yaw, z="HIGH"))
        elif tool == "ALIGN_YAW":
            obj = self.get_obj(args["obj"])
            self.push_gripper(Pose(cell=cur.cell, yaw=obj.yaw, z=cur.z))

    def _apply_user_motion_toward(self, intended_obj_id: str, rng: random.Random) -> None:
        cur = self.gripper_hist[-1]
        intended = self.get_obj(intended_obj_id)
        # Move 1 step toward intended cell with some jitter.
        if rng.random() < 0.8:
            next_cell = gridlib.step_toward(cur.cell, intended.cell)
        else:
            neigh = gridlib.neighbors(cur.cell)
            next_cell = rng.choice(neigh) if neigh else cur.cell
        # Yaw: drift toward object yaw, with occasional oscillation on same cell.
        if cur.cell == intended.cell and rng.random() < 0.55:
            yaw_neighbors = yawlib.neighbors(intended.yaw)
            next_yaw = rng.choice([cur.yaw, yaw_neighbors[0], yaw_neighbors[1]])
        else:
            next_yaw = yawlib.move_toward(cur.yaw, intended.yaw, steps=1)
        # Z: drift down when on cell.
        if next_cell == intended.cell:
            next_z = "LOW" if cur.z != "LOW" and rng.random() < 0.65 else cur.z
        else:
            next_z = "HIGH" if cur.z == "MID" and rng.random() < 0.5 else cur.z
        self.push_gripper(Pose(cell=next_cell, yaw=next_yaw, z=next_z))


class AssistantBackend:
    def predict(self, input_blob: Dict[str, Any], *, world: GridWorld, state: OracleState) -> Dict[str, Any]:
        raise NotImplementedError


class OracleBackend(AssistantBackend):
    def predict(self, input_blob: Dict[str, Any], *, world: GridWorld, state: OracleState) -> Dict[str, Any]:
        return oracle_decide_tool(input_blob["objects"], input_blob["gripper_hist"], input_blob["memory"], state)


class HFBackend(AssistantBackend):
    def __init__(self, cfg: InferenceConfig) -> None:
        self.cfg = cfg

    def predict(self, input_blob: Dict[str, Any], *, world: GridWorld, state: OracleState) -> Dict[str, Any]:
        prompt = f"{INSTRUCTION}\n\nInput:\n{json.dumps(input_blob, ensure_ascii=False)}"
        out = generate_json_only(prompt, self.cfg)
        # Enforce contract.
        validate_tool_call(out)
        return out


def _build_input(world: GridWorld, memory: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "objects": [o.to_record() for o in world.objects],
        "gripper_hist": [p.to_record() for p in world.gripper_hist],
        "memory": memory,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["oracle", "hf"], default="oracle")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter_path", type=str, default=None)
    ap.add_argument("--merged_model_path", type=str, default=None)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_obj", type=int, default=8)
    ap.add_argument("--collision_p", type=float, default=0.2)
    ap.add_argument("--candidate_max_dist", type=int, default=2)
    args = ap.parse_args()

    import tkinter as tk
    from tkinter import ttk

    rng = random.Random(args.seed)

    def new_world() -> Tuple[GridWorld, OracleState, Dict[str, Any]]:
        w = GridWorld(rng, n_obj=args.n_obj, collision_p=args.collision_p)
        st = OracleState(intended_obj_id=w.intended_obj_id)
        mem: Dict[str, Any] = {"n_interactions": 0, "past_dialogs": [], "candidates": [], "last_tool_calls": []}
        mem["candidates"] = w.candidates(args.candidate_max_dist)
        return w, st, mem

    world, state, memory = new_world()

    if args.backend == "oracle":
        backend: AssistantBackend = OracleBackend()
    else:
        cfg = InferenceConfig(
            model_name=args.model_name,
            adapter_path=args.adapter_path,
            merged_model_path=args.merged_model_path,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
        backend = HFBackend(cfg)

    root = tk.Tk()
    root.title("grasp-copilot GUI demo")

    # Layout: left canvas, right controls.
    canvas = tk.Canvas(root, width=520, height=520, bg="white", highlightthickness=1, highlightbackground="#ddd")
    canvas.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="nsew")

    right = ttk.Frame(root)
    right.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=0)
    root.rowconfigure(0, weight=1)

    status = tk.StringVar(value="Ready.")
    ttk.Label(right, textvariable=status, wraplength=360).grid(row=0, column=0, sticky="ew", pady=(0, 8))

    log = tk.Text(right, width=52, height=22)
    log.grid(row=1, column=0, sticky="nsew")
    right.rowconfigure(1, weight=1)

    choices_frame = ttk.Frame(right)
    choices_frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))

    def log_line(s: str) -> None:
        log.insert("end", s + "\n")
        log.see("end")

    def cell_center(cell: str) -> Tuple[int, int]:
        c = gridlib.Cell.from_label(cell)
        margin = 30
        w = int(canvas["width"])
        h = int(canvas["height"])
        cell_w = (w - 2 * margin) / 3.0
        cell_h = (h - 2 * margin) / 3.0
        cx = int(margin + (c.c + 0.5) * cell_w)
        cy = int(margin + (c.r + 0.5) * cell_h)
        return cx, cy

    def redraw() -> None:
        canvas.delete("all")
        w = int(canvas["width"])
        h = int(canvas["height"])
        margin = 30
        # grid lines
        for i in range(4):
            x = margin + i * (w - 2 * margin) / 3.0
            y = margin + i * (h - 2 * margin) / 3.0
            canvas.create_line(x, margin, x, h - margin, fill="#ddd")
            canvas.create_line(margin, y, w - margin, y, fill="#ddd")

        # cell labels
        for cell in gridlib.CELLS:
            cx, cy = cell_center(cell)
            canvas.create_text(cx - 55, cy - 55, text=cell, fill="#999", font=("Arial", 9))

        # objects
        for o in world.objects:
            cx, cy = cell_center(o.cell)
            r = 18
            canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#333", width=2)
            canvas.create_text(cx, cy + 30, text=o.label, fill="#111", font=("Arial", 10))

        # gripper
        g = world.gripper_hist[-1]
        gx, gy = cell_center(g.cell)
        r = 10
        canvas.create_oval(gx - r, gy - r, gx + r, gy + r, fill="#1f77b4", outline="#1f77b4")
        dx, dy = _yaw_dir(g.yaw)
        canvas.create_line(gx, gy, gx + int(dx * 24), gy + int(dy * 24), fill="#1f77b4", width=3)
        canvas.create_text(gx, gy - 22, text=f"{g.yaw}/{g.z}", fill="#1f77b4", font=("Arial", 10, "bold"))

    def clear_choices() -> None:
        for child in list(choices_frame.winfo_children()):
            child.destroy()

    def update_candidates() -> None:
        memory["candidates"] = world.candidates(args.candidate_max_dist)

    def push_user_dialog(text: str) -> None:
        memory["past_dialogs"].append({"role": "user", "content": text})

    def push_assistant_dialog(text: str) -> None:
        memory["past_dialogs"].append({"role": "assistant", "content": text})
        memory["n_interactions"] = int(memory.get("n_interactions", 0)) + 1

    def record_tool_call(tool: str) -> None:
        memory["last_tool_calls"].append(tool)
        memory["last_tool_calls"] = memory["last_tool_calls"][-3:]

    def on_choice_clicked(choice_str: str) -> None:
        # Feed the chosen response back into memory in the same style as the dataset:
        # - YES/NO prompts -> "YES"/"NO"
        # - numbered prompts -> "1", "2", ...
        if choice_str.strip().upper() in {"YES", "NO"}:
            push_user_dialog(choice_str.strip().upper())
        else:
            # If it's "1) foo" style, store the index as a string (e.g. "1").
            prefix = choice_str.split(")", 1)[0].strip()
            push_user_dialog(prefix if prefix.isdigit() else _strip_choice_label(choice_str))
        clear_choices()
        update_candidates()
        redraw()

    def ask_assistance() -> None:
        clear_choices()
        update_candidates()
        input_blob = _build_input(world, memory)
        try:
            tool_call = backend.predict(input_blob, world=world, state=state)
            validate_tool_call(tool_call)
        except Exception as e:
            status.set(f"Model error: {e}")
            log_line(f"[error] {e}")
            return

        record_tool_call(tool_call["tool"])

        log_line(f"[assistant] {json.dumps(tool_call, ensure_ascii=False)}")

        if tool_call["tool"] == "INTERACT":
            text = tool_call["args"]["text"]
            choices: Sequence[str] = tool_call["args"]["choices"]
            status.set(text)
            push_assistant_dialog(text)
            # Render choices as buttons.
            for c in choices:
                ttk.Button(choices_frame, text=c, command=lambda cc=c: on_choice_clicked(cc)).pack(fill="x", pady=2)
            return

        # Motion tools apply immediately.
        status.set(f"Executing: {tool_call['tool']}({tool_call['args']})")
        world.apply_tool(tool_call)
        update_candidates()
        redraw()

    def reset_env() -> None:
        nonlocal world, state, memory
        world, state, memory = new_world()
        clear_choices()
        status.set("Reset.")
        log.delete("1.0", "end")
        redraw()

    def move_gripper_to(cell: str) -> None:
        g = world.gripper_hist[-1]
        world.push_gripper(Pose(cell=cell, yaw=g.yaw, z=g.z))
        update_candidates()
        redraw()

    def rotate_yaw(delta: int) -> None:
        g = world.gripper_hist[-1]
        i = yawlib.YAW_BINS.index(yawlib.normalize(g.yaw))
        n = len(yawlib.YAW_BINS)
        new_yaw = yawlib.YAW_BINS[(i + delta) % n]
        world.push_gripper(Pose(cell=g.cell, yaw=new_yaw, z=g.z))
        update_candidates()
        redraw()

    def change_z(delta: int) -> None:
        g = world.gripper_hist[-1]
        bins = list(Z_BINS)
        i = bins.index(g.z)
        new_z = bins[max(0, min(len(bins) - 1, i + delta))]
        world.push_gripper(Pose(cell=g.cell, yaw=g.yaw, z=new_z))
        update_candidates()
        redraw()

    def on_key(event: Any) -> None:
        key = getattr(event, "keysym", "")
        g = world.gripper_hist[-1]
        c = gridlib.Cell.from_label(g.cell)
        if key == "Left":
            nc = gridlib.Cell(c.r, max(0, c.c - 1)).to_label()
            move_gripper_to(nc)
        elif key == "Right":
            nc = gridlib.Cell(c.r, min(2, c.c + 1)).to_label()
            move_gripper_to(nc)
        elif key == "Up":
            nc = gridlib.Cell(max(0, c.r - 1), c.c).to_label()
            move_gripper_to(nc)
        elif key == "Down":
            nc = gridlib.Cell(min(2, c.r + 1), c.c).to_label()
            move_gripper_to(nc)
        elif key in {"q", "Q"}:
            rotate_yaw(-1)
        elif key in {"e", "E"}:
            rotate_yaw(+1)
        elif key in {"w", "W"}:
            change_z(-1)
        elif key in {"s", "S"}:
            change_z(+1)

    btns = ttk.Frame(root)
    btns.grid(row=1, column=1, padx=10, pady=(0, 10), sticky="ew")
    ttk.Button(btns, text="Ask assistance", command=ask_assistance).pack(side="left", padx=(0, 8))
    ttk.Button(btns, text="Reset", command=reset_env).pack(side="left")

    root.bind("<Key>", on_key)
    redraw()
    log_line(f"[info] backend={args.backend}")
    log_line("[info] Controls: arrows=move, q/e=yaw, w/s=z")
    root.mainloop()


if __name__ == "__main__":
    main()


