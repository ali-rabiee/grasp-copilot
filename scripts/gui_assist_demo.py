"""
Minimal GUI playground to visually validate the grasp-copilot policy.

- White canvas shows a 3x3 grid (A1..C3), object locations + labels, and the gripper pose.
- Use keyboard to move the gripper:
  - Arrow keys: move cell (4-connected)
  - Q/E: rotate yaw (counter-clockwise / clockwise)
  - W/S: change z (HIGH/MID/LOW)
- Click "Ask assistance" to query either:
  - the oracle policy (fast debug), or
  - a HuggingFace model via llm.inference (your trained *merged* model).

Run:
  conda activate talm
  python scripts/gui_assist_demo.py --backend oracle

  # HF backend (example; update paths as needed):
  python scripts/gui_assist_demo.py --backend hf --model_path outputs/qwen_merged_005
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import _bootstrap  # noqa: F401

from data_generator import grid as gridlib
from data_generator import yaw as yawlib
from data_generator.episode import OBJECT_LABELS, Z_BINS, Episode as EpisodeYCB
from data_generator.episode_stacking import CUBE_LABELS, EpisodeStacking
from data_generator.episode_pouring import EpisodePouring, FILL_LEVELS, POUR_AMOUNTS
from data_generator.oracle import OracleState, oracle_decide_tool, validate_tool_call
from data_generator.oracle_pouring import pouring_decide_tool
from data_generator.oracle_registry import ENV_REGISTRY, get_spec
from data_generator.oracle_stacking import stacking_decide_tool
from llm.inference import InferenceConfig, generate_json_only


INSTRUCTION = (
    "Given the robot observation and dialog context, infer the user's intent and "
    "emit exactly one tool call. Output ONLY the tool call JSON with keys tool and args. "
    "If the tool is INTERACT, you must output at most 5 choices total."
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
    """Env-agnostic adapter wrapping a per-env Episode.

    The canvas renderer and the keyboard handler use this class via a small
    surface (objects, gripper_hist, intended_obj_id, get_obj, candidates,
    push_gripper, apply_tool) so they don't need to know whether the
    underlying scene is YCB, cube_stacking, or pouring.
    """

    def __init__(self, env_name: str, rng: random.Random, *, n_obj: int, collision_p: float) -> None:
        self.env_name = env_name
        if env_name == "reach_to_grasp_ycb":
            max_n = min(n_obj, len(OBJECT_LABELS))
            min_n = max(2, min(n_obj, max_n))
            self.episode = EpisodeYCB(
                rng=rng, episode_id=0,
                n_obj=rng.randint(min_n, max_n) if max_n > min_n else min_n,
                collision_p=collision_p,
            )
        elif env_name == "cube_stacking":
            max_n = min(n_obj, len(CUBE_LABELS))
            n_cubes = max(2, min(n_obj, max_n))
            self.episode = EpisodeStacking(rng=rng, episode_id=0, n_cubes=n_cubes)
        elif env_name == "pouring":
            self.episode = EpisodePouring(rng=rng, episode_id=0)
        else:
            raise ValueError(f"Unsupported env: {env_name}")

    # ---- mirror the legacy GridWorld surface ----
    @property
    def objects(self) -> List[Any]:
        return self.episode.objects

    @property
    def gripper_hist(self) -> List[Any]:
        return self.episode.gripper_hist

    @property
    def intended_obj_id(self) -> str:
        return self.episode.intended_obj_id

    def get_obj(self, obj_id: str) -> Any:
        return self.episode.get_obj(obj_id)

    def candidates(self, max_dist: int) -> List[str]:
        return self.episode.gripper_candidates(max_dist=max_dist)

    def push_gripper(self, pose: Pose) -> None:
        # Each per-env Episode has its own Pose dataclass with the same fields.
        # Build the env-correct one to keep type expectations clean.
        if self.env_name == "reach_to_grasp_ycb":
            from data_generator.episode import Pose as EnvPose  # type: ignore
        elif self.env_name == "cube_stacking":
            from data_generator.episode_stacking import Pose as EnvPose  # type: ignore
        else:
            from data_generator.episode_pouring import Pose as EnvPose  # type: ignore
        self.episode._push_gripper(EnvPose(cell=pose.cell, yaw=pose.yaw, z=pose.z))

    def apply_tool(self, tool_call: Dict[str, Any]) -> None:
        self.episode.apply_tool(tool_call)


class AssistantBackend:
    def predict(self, input_blob: Dict[str, Any], *, world: GridWorld, state: OracleState) -> Dict[str, Any]:
        raise NotImplementedError


class OracleBackend(AssistantBackend):
    def __init__(self, env: str = "reach_to_grasp_ycb") -> None:
        self.env = env
        self.decide_fn = get_spec(env).decide_fn

    def predict(self, input_blob: Dict[str, Any], *, world: GridWorld, state: OracleState) -> Dict[str, Any]:
        return self.decide_fn(
            input_blob["objects"],
            input_blob["gripper_hist"],
            input_blob["memory"],
            state,
            user_state=input_blob.get("user_state"),
        )


class HFBackend(AssistantBackend):
    """
    HuggingFace backend that keeps the model loaded across GUI interactions.
    This makes the "Ask assistance" button responsive and avoids repeated VRAM spikes.
    """

    def __init__(self, cfg: InferenceConfig) -> None:
        self.cfg = cfg
        self._loaded = False
        self._model = None
        self._tok = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        from llm.inference import _load_model_and_tokenizer

        self._model, self._tok = _load_model_and_tokenizer(self.cfg)
        self._loaded = True

    def predict(self, input_blob: Dict[str, Any], *, world: GridWorld, state: OracleState) -> Dict[str, Any]:
        self._ensure_loaded()
        assert self._model is not None and self._tok is not None

        def _filter_memory_for_model(mem: Any) -> Any:
            """
            Keep GUI inference input aligned with training schema.
            Training uses memory keys: n_interactions, past_dialogs, candidates, last_tool_calls, excluded_obj_ids, last_action, last_prompt.
            The GUI runtime may add extra keys (e.g., user_state) that the model never saw during training.
            """
            if not isinstance(mem, dict):
                return mem
            keep = {
                "n_interactions",
                "past_dialogs",
                "candidates",
                "last_tool_calls",
                "excluded_obj_ids",
                "last_action",
                "last_prompt",
            }
            return {k: mem.get(k) for k in keep if k in mem}

        def _extract_first_json_object(s: str) -> Optional[str]:
            """
            Best-effort extraction of the first {...} JSON object from a model string.
            Helps when the model outputs valid JSON plus trailing commentary (common with small models).
            """
            if not isinstance(s, str):
                return None
            start = s.find("{")
            if start < 0:
                return None
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(s)):
                ch = s[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                else:
                    if ch == '"':
                        in_str = True
                        continue
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            return s[start : i + 1]
            return None

        # Align input schema with training (important for small models).
        model_input = dict(input_blob)
        model_input["memory"] = _filter_memory_for_model(model_input.get("memory"))

        prompt = f"{INSTRUCTION}\n\nInput:\n{json.dumps(model_input, ensure_ascii=False)}"
        # Reuse the already-loaded model/tokenizer.
        from llm.inference import _build_messages, _generate_once, json_loads_strict

        messages = _build_messages(prompt)
        raw1 = _generate_once(self._model, self._tok, messages, self.cfg)
        try:
            out = json_loads_strict(raw1)
        except Exception:
            # If it's "JSON + extra data", try extracting the first object before asking for repair.
            extracted = _extract_first_json_object(raw1)
            if extracted:
                try:
                    out = json_loads_strict(extracted)
                except Exception:
                    out = None
            else:
                out = None

        if out is None:
            repair_messages = _build_messages("Return ONLY valid JSON for the previous answer.\n\nPrevious answer:\n" + raw1)
            raw2 = _generate_once(self._model, self._tok, repair_messages, self.cfg)
            try:
                out = json_loads_strict(raw2)
            except Exception:
                extracted2 = _extract_first_json_object(raw2)
                if extracted2:
                    out = json_loads_strict(extracted2)
                else:
                    raise

        # Be tolerant of extra keys inside args (models sometimes emit additional metadata).
        # For GUI usage, we strip to the minimal schema before validating.
        if isinstance(out, dict):
            tool = out.get("tool")
            args = out.get("args")
            if tool == "INTERACT" and isinstance(args, dict):
                def _cap_and_renumber_choices(raw_choices: Any, max_total: int = 5) -> List[str]:
                    if not isinstance(raw_choices, list):
                        return []
                    # Keep ordering, but ensure "None of them" (if present) is kept as the last choice.
                    labels: List[str] = []
                    none_label: Optional[str] = None
                    for c in raw_choices:
                        lab = _strip_choice_label(str(c)).strip()
                        if lab.lower() == "none of them":
                            none_label = "None of them"
                        else:
                            labels.append(lab)
                    if none_label is not None:
                        labels = labels[: max_total - 1] + [none_label]
                    else:
                        labels = labels[:max_total]
                    return [f"{i+1}) {lab}" for i, lab in enumerate(labels)]

                capped_choices = _cap_and_renumber_choices(args.get("choices"))
                out = {
                    "tool": "INTERACT",
                    "args": {
                        "kind": args.get("kind"),
                        "text": args.get("text"),
                        "choices": capped_choices,
                    },
                }
            elif tool in {"APPROACH", "ALIGN_YAW"} and isinstance(args, dict):
                out = {"tool": tool, "args": {"obj": args.get("obj")}}
        validate_tool_call(out)
        return out


def _build_input(world: GridWorld, memory: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "objects": [o.to_record() for o in world.objects],
        "gripper_hist": [p.to_record() for p in world.gripper_hist],
        "memory": memory,
        "user_state": memory.get("user_state", {"mode": "translation"}),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["oracle", "hf"], default="oracle")
    ap.add_argument("--env", choices=sorted(ENV_REGISTRY.keys()),
                    default="reach_to_grasp_ycb",
                    help="Which environment / oracle to visually test. "
                         "Defaults to YCB (backward compatible).")
    ap.add_argument("--model_path", type=str, default=None, help="Path/HF id of a merged standalone model (HF backend).")
    # Backward compatible aliases (deprecated).
    ap.add_argument("--model_name", type=str, default=None, help="DEPRECATED: use --model_path")
    ap.add_argument("--merged_model_path", type=str, default=None, help="DEPRECATED: use --model_path")
    ap.add_argument("--adapter_path", type=str, default=None, help="DEPRECATED: adapters are no longer supported here; use merged models.")
    ap.add_argument("--use_4bit", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="Make HF backend as deterministic as possible (forces greedy decoding + deterministic torch settings).",
    )
    ap.add_argument("--n_obj", type=int, default=None,
                    help="Number of objects to spawn. Defaults: ycb=8, cube_stacking=4, pouring=3.")
    ap.add_argument("--collision_p", type=float, default=0.2)
    ap.add_argument("--candidate_max_dist", type=int, default=None,
                    help="Manhattan radius for the candidate set. Defaults from oracle_registry.")
    args = ap.parse_args()

    # Resolve env-aware defaults so YCB scripts keep behaving identically.
    if args.n_obj is None:
        args.n_obj = {"reach_to_grasp_ycb": 8, "cube_stacking": 4, "pouring": 3}[args.env]
    if args.candidate_max_dist is None:
        args.candidate_max_dist = get_spec(args.env).default_candidate_max_dist

    import tkinter as tk
    from tkinter import ttk

    rng = random.Random(args.seed)

    def new_world() -> Tuple[GridWorld, OracleState, Dict[str, Any]]:
        w = GridWorld(args.env, rng, n_obj=args.n_obj, collision_p=args.collision_p)
        st = OracleState(intended_obj_id=w.intended_obj_id)
        mem: Dict[str, Any] = {
            "n_interactions": 0,
            "past_dialogs": [],
            "candidates": [],
            "last_tool_calls": [],
            "excluded_obj_ids": [],
            "last_action": {},
            "user_state": {"mode": "translation"},
        }
        mem["candidates"] = w.candidates(args.candidate_max_dist)
        return w, st, mem

    world, state, memory = new_world()

    if args.backend == "oracle":
        backend: AssistantBackend = OracleBackend(env=args.env)
    else:
        model_path = args.model_path or args.merged_model_path or args.model_name
        if not model_path:
            model_path = "Qwen/Qwen2.5-7B-Instruct"
            print("[gui] WARNING: defaulting to Qwen/Qwen2.5-7B-Instruct; pass --model_path for a merged model.")
        if args.adapter_path:
            raise SystemExit("adapter_path is deprecated and not supported. Please pass a merged model via --model_path.")
        if args.deterministic:
            # Force greedy decoding for stable debugging.
            args.temperature = 0.0
            args.top_p = 1.0

        # If the provided path doesn't exist, HF will treat it as a Hub repo id.
        # Be robust to running from different working directories.
        mp = str(model_path)
        if ("/" in mp or mp.startswith(".")) and not Path(mp).expanduser().exists():
            for prefix in ("./grasp-copilot/", "grasp-copilot/"):
                if mp.startswith(prefix):
                    mp2 = mp[len(prefix) :]
                    if Path(mp2).expanduser().exists():
                        model_path = mp2
                        break
        cfg = InferenceConfig(
            model_path=str(model_path),
            use_4bit=args.use_4bit,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            deterministic=bool(args.deterministic),
        )
        backend = HFBackend(cfg)

    root = tk.Tk()
    root.title(f"grasp-copilot GUI demo · env={args.env} · backend={args.backend}")
    # Maximize by default (cross-platform best-effort).
    # - Windows often supports: root.state("zoomed")
    # - Some Linux window managers support: root.attributes("-zoomed", True)
    # If neither works, fall back to setting geometry to the screen size.
    try:
        root.state("zoomed")
    except Exception:
        try:
            root.attributes("-zoomed", True)
        except Exception:
            try:
                w = root.winfo_screenwidth()
                h = root.winfo_screenheight()
                root.geometry(f"{w}x{h}+0+0")
            except Exception:
                pass

    # Layout: left canvas, right controls.
    canvas = tk.Canvas(root, width=900, height=700, bg="white", highlightthickness=1, highlightbackground="#ddd")
    canvas.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="nsew")

    right = ttk.Frame(root)
    right.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=0)
    root.rowconfigure(0, weight=1)

    status = tk.StringVar(value="Ready.")
    SIDEBAR_FONT = ("Arial", 14)
    SIDEBAR_FONT_BOLD = ("Arial", 14, "bold")
    SIDEBAR_MONO = ("Courier", 12)

    # Make ttk buttons bigger/readable.
    style = ttk.Style(root)
    style.configure("Big.TButton", font=("Arial", 13, "bold"), padding=(14, 10))
    style.configure("Choice.TButton", font=("Arial", 13), padding=(12, 9))
    style.configure("Mode.TRadiobutton", font=("Arial", 12), padding=(6, 4))

    ttk.Label(
        right,
        textvariable=status,
        wraplength=420,
        font=SIDEBAR_FONT_BOLD,
        justify="left",
    ).grid(row=0, column=0, sticky="ew", pady=(0, 10))

    # Explicit user input mode selector (matches user_state.mode).
    mode_var = tk.StringVar(value=str((memory.get("user_state") or {}).get("mode") or "translation"))

    def set_mode(mode: str) -> None:
        if mode not in {"translation", "rotation", "gripper"}:
            mode = "translation"
        mode_var.set(mode)
        memory["user_state"] = {"mode": mode}
        status.set(f"Mode: {mode}")

    mode_box = ttk.LabelFrame(right, text="Input mode")
    mode_box.grid(row=1, column=0, sticky="ew", pady=(0, 10))
    ttk.Radiobutton(mode_box, text="Translation (move)", value="translation", variable=mode_var, style="Mode.TRadiobutton", command=lambda: set_mode("translation")).pack(anchor="w")
    ttk.Radiobutton(mode_box, text="Rotation (yaw)", value="rotation", variable=mode_var, style="Mode.TRadiobutton", command=lambda: set_mode("rotation")).pack(anchor="w")
    ttk.Radiobutton(mode_box, text="Gripper (open/close)", value="gripper", variable=mode_var, style="Mode.TRadiobutton", command=lambda: set_mode("gripper")).pack(anchor="w")

    log = tk.Text(right, width=52, height=22, font=SIDEBAR_MONO)
    log.grid(row=2, column=0, sticky="nsew")
    right.rowconfigure(2, weight=1)

    choices_frame = ttk.Frame(right)
    choices_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))

    def log_line(s: str) -> None:
        log.insert("end", s + "\n")
        log.see("end")

    def cell_center(cell: str) -> Tuple[int, int]:
        c = gridlib.Cell.from_label(cell)
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        # During early init, winfo_* can be 1.
        if w <= 2:
            w = int(canvas["width"])
        if h <= 2:
            h = int(canvas["height"])
        margin = max(18, int(min(w, h) * 0.06))
        cell_w = (w - 2 * margin) / 3.0
        cell_h = (h - 2 * margin) / 3.0
        cx = int(margin + (c.c + 0.5) * cell_w)
        cy = int(margin + (c.r + 0.5) * cell_h)
        return cx, cy

    def redraw() -> None:
        canvas.delete("all")
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w <= 2:
            w = int(canvas["width"])
        if h <= 2:
            h = int(canvas["height"])
        margin = max(18, int(min(w, h) * 0.06))
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
        cell_w = (w - 2 * margin) / 3.0
        cell_h = (h - 2 * margin) / 3.0

        def cell_rect(cell: str) -> Tuple[float, float, float, float]:
            c = gridlib.Cell.from_label(cell)
            x0 = margin + c.c * cell_w
            y0 = margin + c.r * cell_h
            return x0, y0, x0 + cell_w, y0 + cell_h

        objs_by_cell: Dict[str, List[Obj]] = defaultdict(list)
        for o in world.objects:
            objs_by_cell[o.cell].append(o)

        # Predefined layout points within a cell (normalized 0..1).
        layout_pts: List[Tuple[float, float]] = [
            (0.33, 0.33),
            (0.66, 0.33),
            (0.33, 0.66),
            (0.66, 0.66),
            (0.50, 0.33),
            (0.50, 0.66),
            (0.33, 0.50),
            (0.66, 0.50),
            (0.50, 0.50),
        ]

        for cell, objs in objs_by_cell.items():
            x0, y0, x1, y1 = cell_rect(cell)
            # Keep drawing stable but deterministic.
            objs = sorted(objs, key=lambda oo: oo.id)

            n = len(objs)
            base_r = max(10, int(min(w, h) * 0.018))
            r = max(7, int(base_r * (0.85 if n > 1 else 1.0)))

            pts = layout_pts[: min(n, len(layout_pts))]
            for i, o in enumerate(objs):
                px, py = pts[i] if i < len(pts) else (0.5, 0.5)
                cx = x0 + px * (x1 - x0)
                cy = y0 + py * (y1 - y0)
                kind = getattr(o, "kind", None)
                is_held = bool(getattr(o, "is_held", False))
                fill_level = getattr(o, "fill", None)
                top_of_stack = getattr(o, "top_of_stack", True)

                # Pouring: pitcher rendered as a tall rectangle; cup as an open beaker.
                if kind == "pitcher":
                    canvas.create_rectangle(
                        cx - r, cy - int(r * 1.3), cx + r, cy + int(r * 1.0),
                        outline="#7f3d00", width=3, fill="#fff3e0",
                    )
                    # spout
                    canvas.create_polygon(
                        cx + r, cy - int(r * 1.1), cx + int(r * 1.5), cy - int(r * 0.7), cx + r, cy - int(r * 0.6),
                        outline="#7f3d00", fill="#fff3e0", width=2,
                    )
                elif kind == "cup":
                    canvas.create_rectangle(
                        cx - r, cy - r, cx + r, cy + r,
                        outline="#1f5fa8", width=3, fill="#eaf2fb",
                    )
                else:
                    # YCB / cube_stacking
                    outline = "#333"
                    width = 2
                    if is_held:
                        outline = "#d62728"
                        width = 4
                    elif not top_of_stack:
                        # Covered cube — dimmer outline
                        outline = "#aaa"
                    canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                       outline=outline, width=width)
                    if is_held:
                        # Small "HELD" badge above the shape.
                        canvas.create_text(cx, cy - int(r * 1.7),
                                           text="HELD", fill="#d62728",
                                           font=("Arial", max(8, int(r * 0.7)), "bold"))

                # Fill-level bar for pouring vessels.
                if fill_level is not None and fill_level in FILL_LEVELS:
                    frac = {"EMPTY": 0.05, "PARTIAL": 0.5, "FULL": 0.95}[fill_level]
                    bar_h = int((2 * r) * frac)
                    bar_color = "#3f9bff" if kind == "cup" else "#b27a3c"
                    canvas.create_rectangle(
                        cx - int(r * 0.6), cy + r - bar_h,
                        cx + int(r * 0.6), cy + r - 1,
                        fill=bar_color, outline="",
                    )
                    canvas.create_text(cx, cy + r + int(r * 0.6),
                                       text=fill_level, fill="#333",
                                       font=("Arial", max(7, int(r * 0.55))))

            # Render labels as a stacked list inside the cell so they don't overlap each other.
            # Show at most 4, then summarize.
            labels = [o.label for o in objs]
            shown = labels[:4]
            lines = [f"{i+1}) {lab}" for i, lab in enumerate(shown)]
            if len(labels) > len(shown):
                lines.append(f"+{len(labels) - len(shown)} more")
            label_text = "\n".join(lines)
            canvas.create_text(
                x0 + 6,
                y1 - 6,
                text=label_text,
                fill="#111",
                anchor="sw",
                font=("Arial", max(9, int(base_r * 0.55))),
            )

        # gripper
        g = world.gripper_hist[-1]
        gx, gy = cell_center(g.cell)
        r = max(8, int(min(w, h) * 0.014))
        canvas.create_oval(gx - r, gy - r, gx + r, gy + r, fill="#1f77b4", outline="#1f77b4")
        dx, dy = _yaw_dir(g.yaw)
        canvas.create_line(
            gx,
            gy,
            gx + int(dx * (r * 2.4)),
            gy + int(dy * (r * 2.4)),
            fill="#1f77b4",
            width=max(2, int(r * 0.35)),
        )
        canvas.create_text(
            gx,
            gy - int(r * 2.0),
            text=f"{g.yaw}/{g.z}",
            fill="#1f77b4",
            font=("Arial", max(9, int(r * 0.9)), "bold"),
        )

    _redraw_after_id: Optional[str] = None

    def schedule_redraw(_event: Any = None) -> None:
        # Debounce rapid resize events.
        nonlocal _redraw_after_id
        if _redraw_after_id is not None:
            try:
                root.after_cancel(_redraw_after_id)
            except Exception:
                pass
        _redraw_after_id = root.after(50, redraw)

    def clear_choices() -> None:
        for child in list(choices_frame.winfo_children()):
            child.destroy()

    def update_candidates() -> None:
        cands = world.candidates(args.candidate_max_dist)
        ex = set(memory.get("excluded_obj_ids") or [])
        if ex:
            cands = [c for c in cands if c not in ex]
        memory["candidates"] = cands

    def push_user_dialog(text: str) -> None:
        memory["past_dialogs"].append({"role": "user", "content": text})

    def push_assistant_dialog(text: str) -> None:
        memory["past_dialogs"].append({"role": "assistant", "content": text})
        memory["n_interactions"] = int(memory.get("n_interactions", 0)) + 1

    def record_tool_call(tool: str) -> None:
        memory["last_tool_calls"].append(tool)
        memory["last_tool_calls"] = memory["last_tool_calls"][-3:]

    def _choice_to_user_content(choice_str: str) -> str:
        """
        Convert a clicked choice button label into a user content string consistent with the dataset:
        - YES/NO prompts -> "YES"/"NO"
        - numbered prompts -> "1", "2", ...
        """
        s = choice_str.strip()
        # If the semantic label is YES/NO, always return YES/NO (even if it's "1) YES").
        semantic = _strip_choice_label(s).strip().upper()
        if semantic in {"YES", "NO"}:
            return semantic
        # Otherwise, return the semantic label (preferred for training).
        return _strip_choice_label(s).strip()

    def _choice_to_user_content_hf(choice_str: str) -> str:
        # Same behavior as oracle: store semantic labels for training consistency.
        return _choice_to_user_content(choice_str)

    def _apply_none_of_them_exclusion_from_last_prompt() -> None:
        """
        If the user clicked 'None of them', exclude the previously presented object options
        (when we can resolve them to object ids).
        """
        last = memory.get("last_prompt") or {}
        choices = list((last.get("choices") or [])) if isinstance(last, dict) else []
        labels: List[str] = []
        for c in choices:
            lab = _strip_choice_label(str(c)).strip()
            if lab.upper() in {"YES", "NO"} or lab.lower() == "none of them":
                continue
            labels.append(lab)
        if not labels:
            return
        ex = set(memory.get("excluded_obj_ids") or [])
        for lab in labels:
            for o in world.objects:
                if o.label == lab:
                    ex.add(o.id)
        memory["excluded_obj_ids"] = sorted(ex)
        update_candidates()

    def _apply_oracle_user_reply(user_content: str) -> bool:
        """
        Advance the oracle state machine based on the last prompt context and a user reply.
        This mirrors the transitions used by the dataset simulator so the oracle backend
        behaves interactively in the GUI.

        Returns:
            True if the GUI should immediately auto-continue by calling `ask_assistance()`.
        """
        ctx = state.last_prompt_context or {}
        t = ctx.get("type")

        def reset_conversation_only() -> None:
            # Keep the world/gripper as-is, but reset assistant memory + oracle state so
            # the user can ask for assistance again later.
            memory["n_interactions"] = 0
            memory["past_dialogs"] = []
            memory["last_tool_calls"] = []
            memory["excluded_obj_ids"] = []
            memory["last_action"] = {}
            update_candidates()
            state.intended_obj_id = world.intended_obj_id
            state.selected_obj_id = None
            state.pending_action_obj_id = None
            state.pending_mode = None
            state.awaiting_confirmation = False
            state.awaiting_help = False
            state.awaiting_choice = False
            state.awaiting_intent_gate = False
            state.awaiting_anything_else = False
            state.awaiting_mode_select = False
            state.terminate_episode = False
            state.last_prompt_context = None

        def set_selected_by_label(label: str) -> None:
            for o in world.objects:
                if o.label == label:
                    state.selected_obj_id = o.id
                    # Treat selection as the new goal, matching generator behavior.
                    state.intended_obj_id = o.id
                    return

        auto_continue = True

        if t == "intent_gate_candidates":
            if user_content.upper() == "YES":
                state.awaiting_choice = True
                state.awaiting_intent_gate = False
                action = str(ctx.get("action") or "APPROACH").upper()
                state.pending_mode = action if action in {"APPROACH", "ALIGN_YAW"} else "APPROACH"
            else:
                state.awaiting_intent_gate = False
                state.awaiting_choice = False
                state.awaiting_anything_else = True
                state.pending_mode = None
                state.selected_obj_id = None
        elif t == "intent_gate_yaw":
            if user_content.upper() == "YES":
                state.awaiting_help = True
                state.awaiting_intent_gate = False
                state.pending_mode = "ALIGN_YAW"
                obj_id = ctx.get("obj_id")
                if isinstance(obj_id, str):
                    state.selected_obj_id = obj_id
            else:
                state.awaiting_intent_gate = False
                state.awaiting_help = False
                state.awaiting_anything_else = True
                state.pending_mode = None
                state.selected_obj_id = None
        elif t == "candidate_choice":
            labels: List[str] = list(ctx.get("labels") or [])
            obj_ids: List[str] = list(ctx.get("obj_ids") or [])
            none_index = int(ctx.get("none_index") or (len(labels) + 1))
            # Support label replies (preferred for training), numeric replies (backward compat),
            # and "None of them" iterative exclusion.
            if user_content.strip().lower() == "none of them":
                ex = set(memory.get("excluded_obj_ids") or [])
                for oid in obj_ids:
                    ex.add(oid)
                memory["excluded_obj_ids"] = sorted(ex)
                state.selected_obj_id = None
                state.awaiting_choice = True
                state.awaiting_confirmation = False
            elif user_content in labels:
                set_selected_by_label(user_content)
                state.awaiting_choice = False
                state.awaiting_confirmation = False
            elif user_content.isdigit():
                idx = int(user_content) - 1
                if int(user_content) == none_index:
                    ex = set(memory.get("excluded_obj_ids") or [])
                    for oid in obj_ids:
                        ex.add(oid)
                    memory["excluded_obj_ids"] = sorted(ex)
                    state.selected_obj_id = None
                    state.awaiting_choice = True
                    state.awaiting_confirmation = False
                else:
                    if 0 <= idx < len(labels):
                        set_selected_by_label(labels[idx])
                    state.awaiting_choice = False
                    state.awaiting_confirmation = False
        elif t == "confirm":
            obj_id = ctx.get("obj_id")
            action = str(ctx.get("action") or "").upper()
            if user_content.upper() == "YES" and isinstance(obj_id, str):
                state.pending_action_obj_id = obj_id
                state.selected_obj_id = obj_id
                if action in {"APPROACH", "ALIGN_YAW"}:
                    state.pending_mode = action
            else:
                # Reject: go to recovery. Also exclude the declined object so
                # the user isn't re-offered the same target on the next round.
                state.pending_action_obj_id = None
                state.pending_mode = None
                state.selected_obj_id = None
                state.awaiting_anything_else = True
                if isinstance(obj_id, str):
                    ex = set(memory.get("excluded_obj_ids") or [])
                    ex.add(obj_id)
                    memory["excluded_obj_ids"] = sorted(ex)
                    update_candidates()
            state.awaiting_confirmation = False
        elif t == "help":
            obj_id = ctx.get("obj_id")
            if user_content.upper() == "YES" and isinstance(obj_id, str):
                state.pending_action_obj_id = obj_id
                state.selected_obj_id = obj_id
                state.pending_mode = "ALIGN_YAW"
            else:
                state.pending_action_obj_id = None
                state.pending_mode = None
                state.selected_obj_id = None
                state.awaiting_anything_else = True
            state.awaiting_help = False
        elif t == "anything_else":
            if user_content.upper() == "YES":
                # Recovery: re-enter mode_select. Only clear exclusions if the
                # current candidate set would otherwise be empty (e.g., user said
                # "None of them" repeatedly). Otherwise preserve them so previously-
                # declined targets stay out of the next round.
                cur_candidates = list(memory.get("candidates") or [])
                excluded = set(memory.get("excluded_obj_ids") or [])
                live = [c for c in cur_candidates if c not in excluded]
                if not live:
                    memory["excluded_obj_ids"] = []
                state.awaiting_mode_select = True
                state.awaiting_anything_else = False
            else:
                # In the dataset generator we can terminate the episode, but in the GUI we
                # want "NO" to simply end the current assistance session while still letting
                # the user ask again later.
                reset_conversation_only()
                status.set("Okay — no assistance. You can press 'Ask assistance' anytime.")
                auto_continue = False
        elif t == "mode_select":
            # Accept semantic or numeric. Choices vary per env (ctx["actions"]).
            uc = user_content.strip().upper()
            actions: List[str] = list(ctx.get("actions") or ["APPROACH", "ALIGN_YAW"])
            picked: Optional[str] = None
            if uc in {a.upper() for a in actions}:
                picked = uc
            elif user_content.isdigit():
                idx = int(user_content) - 1
                if 0 <= idx < len(actions):
                    picked = actions[idx].upper()
            if picked:
                state.pending_mode = picked
            state.awaiting_mode_select = False
            state.awaiting_choice = True
        elif t == "terminal_ack":
            reset_conversation_only()
            status.set("Okay. You can press 'Ask assistance' anytime.")
            auto_continue = False
        # ---------------------------- cube_stacking / pouring contexts ----------------------------
        elif t in {"intent_gate_stack", "intent_gate_pour"}:
            if user_content.upper() == "YES":
                state.awaiting_choice = True
                state.awaiting_intent_gate = False
                action = str(ctx.get("action") or "APPROACH").upper()
                if action not in {"APPROACH", "ALIGN_YAW", "STACK", "GRAB", "POUR"}:
                    action = "APPROACH"
                state.pending_mode = action
            else:
                state.awaiting_intent_gate = False
                state.awaiting_choice = False
                state.awaiting_anything_else = True
                state.pending_mode = None
                state.selected_obj_id = None
        elif t in {"confirm_stack", "confirm_pour", "confirm_grab", "pitcher_acquisition"}:
            obj_id = ctx.get("obj_id")
            action = str(ctx.get("action") or "").upper()
            if user_content.upper() == "YES" and isinstance(obj_id, str):
                state.pending_action_obj_id = obj_id
                state.selected_obj_id = obj_id
                if action in {"APPROACH", "ALIGN_YAW", "STACK", "GRAB", "POUR"}:
                    state.pending_mode = action
            else:
                state.pending_action_obj_id = None
                state.pending_mode = None
                state.selected_obj_id = None
                state.awaiting_anything_else = True
                # Loop preventer: exclude the declined obj_id so re-entering the
                # candidate flow won't re-offer the same target.
                if isinstance(obj_id, str):
                    ex = set(memory.get("excluded_obj_ids") or [])
                    ex.add(obj_id)
                    memory["excluded_obj_ids"] = sorted(ex)
                    update_candidates()
            state.awaiting_confirmation = False
        elif t in {"non_top_redirect", "cup_full_redirect"}:
            alt_id = ctx.get("alt_obj_id")
            if user_content.upper() == "YES" and isinstance(alt_id, str):
                state.intended_obj_id = alt_id
                state.selected_obj_id = None
                state.pending_action_obj_id = None
                state.pending_mode = None
                state.awaiting_intent_gate = False
                state.awaiting_confirmation = False
            else:
                state.awaiting_intent_gate = False
                state.awaiting_anything_else = True
        elif t == "amount_choice":
            amounts: List[str] = list(ctx.get("amounts") or list(POUR_AMOUNTS))
            none_index = int(ctx.get("none_index") or (len(amounts) + 1))
            uc = user_content.strip().upper()
            if uc in {a.upper() for a in amounts}:
                state.pending_amount = uc
                state.awaiting_amount_choice = False
                state.awaiting_amount_confirmation = True
            elif user_content.isdigit():
                idx = int(user_content)
                if idx == none_index:
                    state.awaiting_amount_choice = False
                    state.awaiting_anything_else = True
                elif 1 <= idx <= len(amounts):
                    state.pending_amount = amounts[idx - 1]
                    state.awaiting_amount_choice = False
                    state.awaiting_amount_confirmation = True
            elif user_content.strip().lower().startswith("none"):
                state.awaiting_amount_choice = False
                state.awaiting_anything_else = True
        elif t == "confirm_amount":
            if user_content.upper() == "YES":
                state.awaiting_amount_confirmation = False
                state.pending_action_obj_id = state.selected_obj_id or state.intended_obj_id
                state.pending_mode = "POUR"
                state.awaiting_confirmation = False
            else:
                state.awaiting_amount_confirmation = False
                state.awaiting_amount_choice = True
                state.pending_amount = None

        # Consume the prompt context once applied.
        state.last_prompt_context = None
        return auto_continue

    def on_choice_clicked(choice_str: str) -> None:
        # Oracle backend expects number strings for option picks (matches generator),
        # HF backend works better with semantic labels.
        if args.backend == "hf":
            user_content = _choice_to_user_content_hf(choice_str)
        else:
            user_content = _choice_to_user_content(choice_str)
        log_line(f"[user] {user_content}")
        push_user_dialog(user_content)

        if args.backend == "oracle":
            auto_continue = _apply_oracle_user_reply(user_content)
        else:
            auto_continue = True
            # For HF backend, maintain exclusions when user clicks "None of them".
            # user_content may be a number ("5") if the choice was "5) None of them".
            semantic = _strip_choice_label(choice_str).strip().lower()
            if semantic == "none of them":
                _apply_none_of_them_exclusion_from_last_prompt()

        clear_choices()
        update_candidates()
        redraw()

        # Auto-continue one step so the user immediately sees the next question/action.
        if auto_continue:
            ask_assistance()

    def ask_assistance() -> None:
        clear_choices()
        update_candidates()
        # Ensure the sent user_state matches the selected mode.
        memory["user_state"] = {"mode": mode_var.get()}
        input_blob = _build_input(world, memory)
        try:
            tool_call = backend.predict(input_blob, world=world, state=state)
            validate_tool_call(tool_call, env=args.env)
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
            # Store the last prompt + choices in memory for better interactive behavior.
            memory["last_prompt"] = {"kind": tool_call["args"]["kind"], "text": text, "choices": list(choices)}

            # UI policy: show at most 4 object options (plus optional "None of them").
            # If the model returns more, truncate for readability.
            obj_opts: List[str] = []
            none_opt: Optional[str] = None
            for c in list(choices):
                lab = _strip_choice_label(str(c)).strip().lower()
                if lab == "none of them":
                    none_opt = str(c)
                elif lab in {"yes", "no"}:
                    obj_opts.append(str(c))
                else:
                    obj_opts.append(str(c))
            # Only truncate for non-YES/NO menus (heuristic: contains some non-yes/no labels).
            is_yes_no_only = all(_strip_choice_label(str(c)).strip().upper() in {"YES", "NO"} for c in choices)
            if not is_yes_no_only:
                # Separate object labels from yes/no, then cap objects at 4.
                yes_no = [c for c in obj_opts if _strip_choice_label(str(c)).strip().upper() in {"YES", "NO"}]
                objs = [c for c in obj_opts if _strip_choice_label(str(c)).strip().upper() not in {"YES", "NO"} and _strip_choice_label(str(c)).strip().lower() != "none of them"]
                objs = objs[:4]
                new_choices: List[str] = yes_no + objs
                if none_opt is not None:
                    new_choices.append(none_opt)
                choices = new_choices
            # Render choices as buttons.
            for c in choices:
                ttk.Button(
                    choices_frame,
                    text=c,
                    command=lambda cc=c: on_choice_clicked(cc),
                    style="Choice.TButton",
                ).pack(fill="x", pady=3)
            return

        # Motion tools apply immediately.
        status.set(f"Executing: {tool_call['tool']}({tool_call['args']})")
        world.apply_tool(tool_call)
        motion_set = {"APPROACH", "ALIGN_YAW", "STACK", "GRAB", "POUR", "RELEASE"}
        if tool_call["tool"] in motion_set:
            la: Dict[str, Any] = {"tool": tool_call["tool"]}
            if "obj" in (tool_call.get("args") or {}):
                la["obj"] = tool_call["args"]["obj"]
            if "amount" in (tool_call.get("args") or {}):
                la["amount"] = tool_call["args"]["amount"]
            memory["last_action"] = la
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
        if key in {"1", "2", "3"}:
            set_mode({"1": "translation", "2": "rotation", "3": "gripper"}[key])
            return

        mode = mode_var.get()
        if key in {"Left", "Right", "Up", "Down"}:
            if mode == "translation":
                if key == "Left":
                    nc = gridlib.Cell(c.r, max(0, c.c - 1)).to_label()
                elif key == "Right":
                    nc = gridlib.Cell(c.r, min(2, c.c + 1)).to_label()
                elif key == "Up":
                    nc = gridlib.Cell(max(0, c.r - 1), c.c).to_label()
                else:  # Down
                    nc = gridlib.Cell(min(2, c.r + 1), c.c).to_label()
                move_gripper_to(nc)
            elif mode == "rotation":
                # Rotate yaw with arrows (left=CCW, right=CW). Up/Down also rotate for convenience.
                if key in {"Left", "Down"}:
                    rotate_yaw(-1)
                else:
                    rotate_yaw(+1)
            else:  # gripper
                # Open/close gripper with arrows (up=open -> HIGH, down=close -> LOW).
                if key == "Up":
                    change_z(-1)
                elif key == "Down":
                    change_z(+1)
                # Left/Right no-op in gripper mode.
            return

        # Keep the legacy hotkeys as shortcuts (also set mode accordingly).
        if key in {"q", "Q"}:
            set_mode("rotation")
            rotate_yaw(-1)
        elif key in {"e", "E"}:
            set_mode("rotation")
            rotate_yaw(+1)
        elif key in {"w", "W"}:
            set_mode("gripper")
            change_z(-1)
        elif key in {"s", "S"}:
            set_mode("gripper")
            change_z(+1)

    btns = ttk.Frame(root)
    btns.grid(row=1, column=1, padx=10, pady=(0, 10), sticky="ew")
    ttk.Button(btns, text="Ask assistance", command=ask_assistance, style="Big.TButton").pack(side="left", padx=(0, 8))
    ttk.Button(btns, text="Reset", command=reset_env, style="Big.TButton").pack(side="left")

    root.bind("<Key>", on_key)
    canvas.bind("<Configure>", schedule_redraw)
    redraw()
    from data_generator.oracle import ENV_SKILLS as _ENV_SKILLS
    log_line(f"[info] env={args.env} backend={args.backend}")
    log_line(f"[info] skills available: {list(_ENV_SKILLS[args.env])}")
    log_line("[info] Modes: 1=translation, 2=rotation, 3=gripper")
    log_line("[info] Controls: arrows act in current mode; q/e rotate; w/s open/close")
    root.mainloop()


if __name__ == "__main__":
    main()


