"""Tkinter GUI for the PRIME wizard.

Layout (single 1100x740 window):

  ┌─────────────────────────────────────────────────────────────────┐
  │ Header: episode/tick/mode + alert banner                        │
  ├──────────────────────────────┬──────────────────────────────────┤
  │ Grid panel (matplotlib)      │ Memory panel                     │
  ├──────────────────────────────┴──────────────────────────────────┤
  │ Decision panel: INTERACT form + EXECUTE radio + Submit/Skip    │
  └─────────────────────────────────────────────────────────────────┘

The runner's ``decide`` callback is bound to the Submit button: it returns
the validated tool-call dict and unblocks the runner thread. We use a
thread-safe ``queue.Queue`` to hand the decision back, and ``after()`` on the
Tk root to advance episodes without blocking the UI thread.
"""

from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, List, Optional

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from data_generator.oracle import MAX_INTERACT_CHOICES, validate_tool_call

from ..driver.alert_scheduler import AlertReason
from ..driver.episode_runner import EpisodeRunner, RunnerConfig
from ..io.writer import EpisodeWriter
from .grid_view import GridView


KIND_OPTIONS = ("QUESTION", "CONFIRM", "SUGGESTION")


class WizardApp:
    """Top-level Tk app. Owns the runner and runs episodes in a worker thread."""

    def __init__(self, root: tk.Tk, runner_cfg: RunnerConfig, writer: EpisodeWriter,
                 num_episodes: int = 1):
        self.root = root
        self.root.title(f"PRIME Wizard · {runner_cfg.wizard_id}")
        self.root.geometry("1140x780")

        self.num_episodes = num_episodes
        self.completed_episodes = 0
        self.writer = writer

        self._decision_queue: queue.Queue[Dict] = queue.Queue(maxsize=1)
        self._current_blob: Optional[Dict] = None
        self._current_reason: Optional[AlertReason] = None

        self._build_ui()

        self.runner = EpisodeRunner(runner_cfg, writer, self._decision_callback)
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(50, self._poll_pending_alert)

    # ------------------------------------------------------------ UI build

    def _build_ui(self) -> None:
        # Header
        self.header = tk.Frame(self.root, bg="#222222", padx=10, pady=6)
        self.header.pack(fill="x")
        self.header_label = tk.Label(self.header,
                                     text="Waiting for first alert…",
                                     fg="white", bg="#222222",
                                     font=("Helvetica", 11, "bold"))
        self.header_label.pack(side="left")
        self.alert_banner = tk.Label(self.header, text="", fg="#ffd166",
                                     bg="#222222", font=("Helvetica", 11, "bold"))
        self.alert_banner.pack(side="right")

        # Mid section: grid + memory side by side
        mid = tk.Frame(self.root)
        mid.pack(fill="both", expand=True, padx=8, pady=6)

        left = tk.Frame(mid)
        left.pack(side="left", fill="both", expand=True)
        self.grid_view = GridView()
        self.canvas = FigureCanvasTkAgg(self.grid_view.fig, master=left)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        right = tk.Frame(mid, width=420)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        ttk.Label(right, text="Memory", font=("Helvetica", 11, "bold")).pack(anchor="w", pady=(0, 4))
        self.memory_text = tk.Text(right, width=52, height=20, font=("Courier", 9),
                                   wrap="word", state="disabled", bg="#fafafa")
        self.memory_text.pack(fill="both", expand=True)

        # Decision panel
        decision = tk.Frame(self.root, padx=8, pady=8, bg="#f4f4f4")
        decision.pack(fill="x")

        # INTERACT form
        interact_frame = ttk.LabelFrame(decision, text="Interaction skill")
        interact_frame.pack(side="left", fill="both", expand=True, padx=(0, 6))

        ttk.Label(interact_frame, text="kind:").grid(row=0, column=0, sticky="w")
        self.kind_var = tk.StringVar(value=KIND_OPTIONS[0])
        for i, k in enumerate(KIND_OPTIONS):
            ttk.Radiobutton(interact_frame, text=k, variable=self.kind_var, value=k).grid(row=0, column=1 + i, sticky="w")

        ttk.Label(interact_frame, text="text:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.text_var = tk.StringVar()
        ttk.Entry(interact_frame, textvariable=self.text_var, width=64).grid(row=1, column=1, columnspan=4, sticky="we", pady=(4, 0))

        self.choice_vars: List[tk.StringVar] = []
        for i in range(MAX_INTERACT_CHOICES):
            ttk.Label(interact_frame, text=f"opt {i + 1}:").grid(row=2 + i, column=0, sticky="w", pady=(2, 0))
            v = tk.StringVar()
            self.choice_vars.append(v)
            ttk.Entry(interact_frame, textvariable=v, width=64).grid(row=2 + i, column=1, columnspan=4, sticky="we", pady=(2, 0))

        # EXECUTE form
        execute_frame = ttk.LabelFrame(decision, text="Execution skill")
        execute_frame.pack(side="left", fill="y", padx=(6, 0))

        self.exec_var = tk.StringVar(value="")
        self.exec_listbox = tk.Listbox(execute_frame, height=10, width=36, font=("Courier", 10))
        self.exec_listbox.pack(padx=4, pady=4)

        # Submit row
        submit_row = tk.Frame(self.root, pady=6)
        submit_row.pack(fill="x")
        ttk.Button(submit_row, text="Submit INTERACT", command=self._on_submit_interact).pack(side="left", padx=6)
        ttk.Button(submit_row, text="Submit EXECUTION", command=self._on_submit_execution).pack(side="left", padx=6)
        ttk.Button(submit_row, text="Defer (skip this alert)", command=self._on_skip).pack(side="left", padx=6)
        self.status_label = tk.Label(submit_row, text="", fg="#a00000")
        self.status_label.pack(side="right", padx=8)

    # -------------------------------------------------------- runner thread

    def _worker_loop(self) -> None:
        for _ in range(self.num_episodes):
            self.runner.run_episode()
            self.completed_episodes += 1
        self.writer.close()
        self.root.after(0, self._on_all_done)

    def _decision_callback(self, blob: Dict, reason: AlertReason) -> Dict:
        """Called from the runner thread. Pushes the alert to the UI thread,
        then blocks on the wizard's response."""
        self._current_blob = blob
        self._current_reason = reason
        self.root.after(0, self._render_alert)
        decision = self._decision_queue.get()
        return decision

    def _render_alert(self) -> None:
        blob = self._current_blob or {}
        reason = self._current_reason

        mode = blob.get("user_state", {}).get("mode", "?")
        ep_idx = self.runner.env.episode_idx
        tick = self.runner.env.tick
        self.header_label.config(text=f"Episode {ep_idx + 1}/{self.num_episodes} · tick {tick} · mode={mode}")
        self.alert_banner.config(text=f"⚠ ALERT · {reason.value if reason else ''}")

        # Render grid.
        self.grid_view.render(blob)
        self.canvas.draw()

        # Render memory.
        mem = blob.get("memory", {})
        hist = blob.get("gripper_hist") or []
        hist_str = " → ".join(f"{g.get('cell')}/{g.get('yaw')}" for g in hist) if hist else "(empty)"
        lines = [
            f"gripper_hist (old→new): {hist_str}",
            f"candidates           : {mem.get('candidates')}",
            f"excluded_obj_ids     : {mem.get('excluded_obj_ids')}",
            f"last_action          : {mem.get('last_action')}",
            f"last_tool_calls (3)  : {mem.get('last_tool_calls')}",
            f"last_prompt          : {json.dumps(mem.get('last_prompt') or {}, indent=2)}",
            "",
            "past_dialogs:",
        ]
        for d in mem.get("past_dialogs") or []:
            lines.append(f"  · [{d.get('kind')}] {d.get('prompt')!r}")
            lines.append(f"        choices={d.get('choices')}  reply={d.get('reply')!r}")

        self.memory_text.config(state="normal")
        self.memory_text.delete("1.0", "end")
        self.memory_text.insert("1.0", "\n".join(lines))
        self.memory_text.config(state="disabled")

        # Refresh execution skill listbox (one per non-held object × {APPROACH, ALIGN_YAW}).
        self.exec_listbox.delete(0, "end")
        self._exec_options: List[Dict] = []
        for o in blob.get("objects") or []:
            if o.get("is_held"):
                continue
            for tool in ("APPROACH", "ALIGN_YAW"):
                opt = {"tool": tool, "args": {"obj": o["id"]}}
                self._exec_options.append(opt)
                self.exec_listbox.insert("end", f"{tool:9s} {o['id']}  ({o['label']} @ {o['cell']}, yaw={o['yaw']})")

        # Reset INTERACT form.
        self.text_var.set("")
        for v in self.choice_vars:
            v.set("")
        self.status_label.config(text="")

    def _poll_pending_alert(self) -> None:
        # Used to keep mainloop responsive; nothing to do here unless extended.
        self.root.after(80, self._poll_pending_alert)

    # ----------------------------------------------------------- submit ops

    def _on_submit_interact(self) -> None:
        choices = [v.get().strip() for v in self.choice_vars if v.get().strip()]
        # Auto-prefix numbering if the wizard forgot it.
        normalized: List[str] = []
        for i, c in enumerate(choices):
            if not c.split(")", 1)[0].isdigit():
                c = f"{i + 1}) {c}"
            normalized.append(c)
        tool_call = {
            "tool": "INTERACT",
            "args": {
                "kind": self.kind_var.get(),
                "text": self.text_var.get().strip(),
                "choices": normalized,
            },
        }
        try:
            validate_tool_call(tool_call)
        except Exception as e:
            self.status_label.config(text=f"Invalid INTERACT: {e}")
            return
        self._decision_queue.put(tool_call)

    def _on_submit_execution(self) -> None:
        sel = self.exec_listbox.curselection()
        if not sel:
            self.status_label.config(text="Pick a row in the Execution panel.")
            return
        opt = self._exec_options[sel[0]]
        try:
            validate_tool_call(opt)
        except Exception as e:
            self.status_label.config(text=f"Invalid execution: {e}")
            return
        self._decision_queue.put(opt)

    def _on_skip(self) -> None:
        # A "defer" still has to be a valid tool call; we represent it as a
        # benign SUGGESTION ack so the schema is preserved.
        self._decision_queue.put({
            "tool": "INTERACT",
            "args": {"kind": "SUGGESTION", "text": "[deferred]", "choices": ["1) OK"]},
        })

    # ----------------------------------------------------------- lifecycle

    def _on_all_done(self) -> None:
        messagebox.showinfo("Done", f"Collected {self.completed_episodes} episodes.")
        self.root.destroy()

    def _on_close(self) -> None:
        if messagebox.askokcancel("Quit", "Stop and write out partial data?"):
            self.writer.close()
            self.root.destroy()
