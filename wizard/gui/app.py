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
from ..prompt_factory import (
    build_prompt,
    display_label,
    list_prompt_types,
    precondition_status,
    signature as prompt_signature,
    valid_actions,
    valid_amounts,
    valid_targets,
)
from .grid_view import GridView


KIND_OPTIONS = ("QUESTION", "CONFIRM", "SUGGESTION")

# ─── visual palette ──────────────────────────────────────────────────────
COL_BG          = "#f5f6f8"
COL_PANEL       = "#ffffff"
COL_HEADER_BG   = "#1f2933"
COL_HEADER_FG   = "#ffffff"
COL_MUTED       = "#6b7280"
COL_LABEL       = "#374151"
COL_VALUE       = "#111827"
COL_ACCENT      = "#1a73e8"
COL_BORDER      = "#d1d5db"
COL_LATEST      = "#f9ab00"
COL_ALERT_BG    = "#ffb74d"

REPLY_STYLE = {
    "YES":  ("#d4edda", "#0f5132"),  # green
    "NO":   ("#f8d7da", "#842029"),  # red
    "PICK": ("#cfe2ff", "#084298"),  # blue
    "NEUT": ("#e9ecef", "#212529"),  # gray
}

KIND_CHIP = {
    "QUESTION":   ("#e3f2fd", "#1565c0"),
    "CONFIRM":    ("#fff3e0", "#e65100"),
    "SUGGESTION": ("#f3e5f5", "#6a1b9a"),
}

ALERT_LABEL = {
    "episode_start":    "🆕  EPISODE START",
    "post_execution":   "✅  POST-EXECUTION",
    "candidate_change": "🔄  CANDIDATE SET CHANGED",
    "stochastic":       "🎲  STOCHASTIC ALERT",
    "replay":           "▶  REPLAY DECISION POINT",
}

FONT_BASE     = ("Helvetica", 10)
FONT_SMALL    = ("Helvetica", 9)
FONT_MUTED    = ("Helvetica", 9, "italic")
FONT_LABEL    = ("Helvetica", 10, "bold")
FONT_SECTION  = ("Helvetica", 12, "bold")
FONT_HEADER   = ("Helvetica", 13, "bold")
FONT_CHIP     = ("Helvetica", 9, "bold")
FONT_REPLY    = ("Helvetica", 15, "bold")
FONT_REPLY_HI = ("Helvetica", 20, "bold")


def _classify_reply(reply: str) -> tuple[str, str]:
    r = (reply or "").strip().upper()
    if not r:
        return REPLY_STYLE["NEUT"]
    if r in ("YES", "Y", "ACK", "OK", "CONFIRM"):
        return REPLY_STYLE["YES"]
    if r in ("NO", "N", "NACK", "STOP", "CANCEL"):
        return REPLY_STYLE["NO"]
    # Looks like a multiple-choice pick: "1) ...", "2) ...", or starts with digit
    if r[0].isdigit() or ")" in r[:3]:
        return REPLY_STYLE["PICK"]
    return REPLY_STYLE["NEUT"]


class WizardApp:
    """Top-level Tk app. Owns the runner and runs episodes in a worker thread."""

    def __init__(self, root: tk.Tk, runner_cfg: Optional[RunnerConfig] = None,
                 writer: Optional[EpisodeWriter] = None, num_episodes: int = 1,
                 replay_cfg=None):
        self.root = root
        # Determine env + mode based on which config was provided.
        if replay_cfg is not None:
            self.env = replay_cfg.env_name
            self.mode = "replay"
            wizard_id = replay_cfg.wizard_id
        elif runner_cfg is not None:
            self.env = runner_cfg.env_cfg.env_name
            self.mode = "live"
            wizard_id = runner_cfg.wizard_id
        else:
            raise ValueError("Need either runner_cfg (live mode) or replay_cfg (replay mode)")

        self.root.title(f"PRIME Wizard · {self.mode}({self.env}) · {wizard_id}")
        self.root.geometry("1180x860")

        self.num_episodes = num_episodes
        self.completed_episodes = 0
        self.decisions_submitted = 0
        self.writer = writer

        self._decision_queue: queue.Queue[Dict] = queue.Queue(maxsize=1)
        self._current_blob: Optional[Dict] = None
        self._current_reason: Optional[AlertReason] = None

        self._build_ui()

        if replay_cfg is not None:
            from ..replay import ReplayRunner
            self.runner = ReplayRunner(replay_cfg, writer, self._decision_callback)
            self.total_replay_records = self.runner.num_records()
            self.total_blocking_records = self.runner.num_blocking_records()
            self.header_label.config(
                text=f"replay · 0/{self.total_blocking_records} wizard decisions "
                     f"({self.total_replay_records} records total)"
            )
        else:
            self.runner = EpisodeRunner(runner_cfg, writer, self._decision_callback)
            self.total_replay_records = None
            self.total_blocking_records = None

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

        # Decision panel — templated wizard form
        decision = tk.Frame(self.root, padx=8, pady=8, bg="#f4f4f4")
        decision.pack(fill="x")

        # ============ INTERACT (templated) ============
        interact_frame = ttk.LabelFrame(decision, text="Interaction (templated)")
        interact_frame.pack(side="left", fill="both", expand=True, padx=(0, 6))

        # Prompt type dropdown with friendly labels.
        ttk.Label(interact_frame, text="Prompt:").grid(row=0, column=0, sticky="w")
        self.prompt_type_var = tk.StringVar()
        self._prompt_type_labels: List[str] = [
            display_label(pt) for pt in list_prompt_types(self.env)
        ]
        self._prompt_type_keys: List[str] = list(list_prompt_types(self.env))
        self.prompt_type_combo = ttk.Combobox(
            interact_frame, textvariable=self.prompt_type_var, state="readonly",
            values=self._prompt_type_labels, width=80,
        )
        self.prompt_type_combo.grid(row=0, column=1, columnspan=3, sticky="we", pady=2)
        self.prompt_type_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_form_changed())

        # Target dropdown (visible for prompts that need a target)
        ttk.Label(interact_frame, text="Target:").grid(row=1, column=0, sticky="w")
        self.target_var = tk.StringVar()
        self.target_combo = ttk.Combobox(interact_frame, textvariable=self.target_var, state="readonly", width=40)
        self.target_combo.grid(row=1, column=1, columnspan=3, sticky="we", pady=2)
        self.target_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_form_changed())

        # Alt-target dropdown (only for *_redirect prompts)
        ttk.Label(interact_frame, text="Alt target:").grid(row=2, column=0, sticky="w")
        self.alt_target_var = tk.StringVar()
        self.alt_target_combo = ttk.Combobox(interact_frame, textvariable=self.alt_target_var, state="readonly", width=40)
        self.alt_target_combo.grid(row=2, column=1, columnspan=3, sticky="we", pady=2)
        self.alt_target_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_form_changed())

        # Action dropdown (for plain "confirm")
        ttk.Label(interact_frame, text="Action:").grid(row=3, column=0, sticky="w")
        self.action_var = tk.StringVar()
        self.action_combo = ttk.Combobox(
            interact_frame, textvariable=self.action_var, state="readonly",
            values=list(valid_actions(self.env, "confirm")), width=14,
        )
        self.action_combo.grid(row=3, column=1, sticky="w", pady=2)
        self.action_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_form_changed())

        # Amount dropdown (for confirm_amount)
        ttk.Label(interact_frame, text="Amount:").grid(row=3, column=2, sticky="e")
        self.amount_var = tk.StringVar()
        self.amount_combo = ttk.Combobox(
            interact_frame, textvariable=self.amount_var, state="readonly",
            values=list(valid_amounts()), width=10,
        )
        self.amount_combo.grid(row=3, column=3, sticky="w", pady=2)
        self.amount_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_form_changed())

        # Free-text fallback (only enabled when prompt type = "other")
        ttk.Label(interact_frame, text="Free text:").grid(row=4, column=0, sticky="w")
        self.free_text_var = tk.StringVar()
        self.free_text_entry = ttk.Entry(interact_frame, textvariable=self.free_text_var, width=58, state="disabled")
        self.free_text_entry.grid(row=4, column=1, columnspan=3, sticky="we", pady=2)
        self.free_text_entry.bind("<KeyRelease>", lambda _e: self._on_form_changed())

        self.free_choice_vars: List[tk.StringVar] = []
        self.free_choice_entries: List[ttk.Entry] = []
        for i in range(MAX_INTERACT_CHOICES):
            ttk.Label(interact_frame, text=f"opt {i+1}:").grid(row=5 + i, column=0, sticky="w")
            v = tk.StringVar()
            self.free_choice_vars.append(v)
            ent = ttk.Entry(interact_frame, textvariable=v, width=58, state="disabled")
            ent.grid(row=5 + i, column=1, columnspan=3, sticky="we", pady=1)
            ent.bind("<KeyRelease>", lambda _e: self._on_form_changed())
            self.free_choice_entries.append(ent)

        # Live preview pane
        ttk.Label(interact_frame, text="Preview:").grid(row=10, column=0, sticky="nw", pady=(6, 0))
        self.preview_text = tk.Text(interact_frame, height=12, width=58, font=("Courier", 9),
                                    bg="#ffffff", state="disabled", wrap="word")
        self.preview_text.grid(row=10, column=1, columnspan=3, sticky="we", pady=(6, 0))

        # ============ EXECUTE ============
        execute_frame = ttk.LabelFrame(decision, text="Execution")
        execute_frame.pack(side="left", fill="y", padx=(6, 0))

        ttk.Label(execute_frame, text="Tool:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.exec_tool_var = tk.StringVar()
        self.exec_tool_combo = ttk.Combobox(
            execute_frame, textvariable=self.exec_tool_var, state="readonly", width=14,
        )
        self.exec_tool_combo.grid(row=0, column=1, sticky="w", padx=4, pady=4)
        self.exec_tool_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_form_changed())

        ttk.Label(execute_frame, text="Target:").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        self.exec_target_var = tk.StringVar()
        self.exec_target_combo = ttk.Combobox(
            execute_frame, textvariable=self.exec_target_var, state="readonly", width=32,
        )
        self.exec_target_combo.grid(row=1, column=1, sticky="w", padx=4, pady=4)
        self.exec_target_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_form_changed())

        ttk.Label(execute_frame, text="Amount:").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        self.exec_amount_var = tk.StringVar()
        self.exec_amount_combo = ttk.Combobox(
            execute_frame, textvariable=self.exec_amount_var, state="readonly",
            values=list(valid_amounts()), width=10,
        )
        self.exec_amount_combo.grid(row=2, column=1, sticky="w", padx=4, pady=4)
        self.exec_amount_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_form_changed())

        # ============ Submit row ============
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
        self.decisions_submitted += 1
        return decision

    def _render_alert(self) -> None:
        blob = self._current_blob or {}
        reason = self._current_reason

        mode = blob.get("user_state", {}).get("mode", "?")
        ep_idx = self.runner.env.episode_idx
        tick = self.runner.env.tick
        if self.mode == "replay" and self.total_blocking_records:
            done = self.decisions_submitted + 1
            self.header_label.config(
                text=f"replay · wizard decision {done}/{self.total_blocking_records} · "
                     f"episode {ep_idx+1} · tick {tick} · user_mode={mode}"
            )
        else:
            self.header_label.config(
                text=f"Episode {ep_idx + 1}/{self.num_episodes} · tick {tick} · mode={mode}"
            )
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

        # Refresh execution tool dropdown for this env.
        from data_generator.oracle import ENV_SKILLS
        all_skills = ENV_SKILLS[self.env]
        exec_tools = [s for s in all_skills if s != "INTERACT"]
        self.exec_tool_combo["values"] = exec_tools

        # Reset all form widgets, then re-populate dynamically based on blob.
        self.prompt_type_var.set("")
        self.target_var.set("")
        self.alt_target_var.set("")
        self.action_var.set("")
        self.amount_var.set("")
        self.exec_tool_var.set("")
        self.exec_target_var.set("")
        self.exec_amount_var.set("")
        self.free_text_var.set("")
        for v in self.free_choice_vars:
            v.set("")
        self._on_form_changed()  # refresh dropdown contents + preview
        self.status_label.config(text="")

    def _poll_pending_alert(self) -> None:
        # Used to keep mainloop responsive; nothing to do here unless extended.
        self.root.after(80, self._poll_pending_alert)

    # ----------------------------------------------------------- submit ops

    # ----------------------------------------------------------- form helpers

    def _selected_prompt_type(self) -> Optional[str]:
        """Map the friendly-label dropdown selection back to the prompt-type key."""
        v = self.prompt_type_var.get()
        if not v:
            return None
        try:
            idx = self._prompt_type_labels.index(v)
        except ValueError:
            # Wizard typed something; allow if it matches a key directly.
            return v if v in self._prompt_type_keys else None
        return self._prompt_type_keys[idx]

    def _target_label(self, obj: Dict) -> str:
        """Render an object as a dropdown entry."""
        extras = []
        if obj.get("is_held"): extras.append("HELD")
        if obj.get("kind"): extras.append(obj["kind"])
        if obj.get("fill"): extras.append(f"fill={obj['fill']}")
        if obj.get("top_of_stack") is False: extras.append("covered")
        suffix = f" [{', '.join(extras)}]" if extras else ""
        return f"{obj['id']}  {obj.get('label','?')} @ {obj.get('cell','?')}, yaw={obj.get('yaw','?')}{suffix}"

    def _id_from_target_label(self, label: str) -> Optional[str]:
        return label.split()[0] if label else None

    def _on_form_changed(self) -> None:
        """Re-derive widget visibility, target dropdown contents, and preview pane."""
        blob = self._current_blob or {}
        objs = blob.get("objects") or []
        if not objs:
            return
        pt = self._selected_prompt_type() or ""

        # Interaction target list filtered by prompt type.
        if pt:
            int_targets = valid_targets(self.env, pt, blob)
            self.target_combo["values"] = [self._target_label(o) for o in int_targets]
            # Alt target gets the same pool, minus the chosen primary.
            primary_id = self._id_from_target_label(self.target_var.get())
            alt_pool = [o for o in int_targets if o["id"] != primary_id]
            self.alt_target_combo["values"] = [self._target_label(o) for o in alt_pool]

        # Show/hide action/amount based on prompt signature.
        sig = set(prompt_signature(pt))
        self._set_widget_state(self.action_combo, "action" in sig)
        self._set_widget_state(self.amount_combo, "amount" in sig)
        self._set_widget_state(self.alt_target_combo, "alt_target" in sig)
        self._set_widget_state(self.free_text_entry, pt == "other")
        for ent in self.free_choice_entries:
            self._set_widget_state(ent, pt == "other")

        # Execution target list filtered by tool.
        tool = self.exec_tool_var.get()
        if tool:
            pool = self._exec_targets_for_tool(blob, tool)
            self.exec_target_combo["values"] = [self._target_label(o) for o in pool]
            self._set_widget_state(self.exec_amount_combo, tool == "POUR")

        # Live preview.
        self._refresh_preview()

    @staticmethod
    def _set_widget_state(widget, enabled: bool) -> None:
        try:
            widget.config(state="readonly" if enabled and "combobox" in widget.winfo_class().lower() else ("normal" if enabled else "disabled"))
        except Exception:
            try:
                widget.config(state="normal" if enabled else "disabled")
            except Exception:
                pass

    def _exec_targets_for_tool(self, blob: Dict, tool: str) -> List[Dict]:
        objs = blob.get("objects") or []
        if tool == "GRAB":
            return [o for o in objs if o.get("kind") == "pitcher" and not o.get("is_held")]
        if tool == "POUR":
            return [o for o in objs if o.get("kind") == "cup" and o.get("fill") != "FULL"]
        if tool == "STACK":
            return [o for o in objs if not o.get("is_held") and o.get("top_of_stack", True)]
        if tool == "RELEASE":
            return []  # no target
        # APPROACH / ALIGN_YAW: any non-held object
        return [o for o in objs if not o.get("is_held")]

    def _build_interact_from_form(self) -> Optional[Dict]:
        pt = self._selected_prompt_type()
        if not pt:
            self.status_label.config(text="Pick a prompt type.")
            return None
        kwargs: Dict = {}
        sig = set(prompt_signature(pt))
        if "target" in sig:
            tid = self._id_from_target_label(self.target_var.get())
            if not tid:
                self.status_label.config(text=f"Prompt {pt!r} needs a target.")
                return None
            kwargs["target_id"] = tid
        if "alt_target" in sig:
            atid = self._id_from_target_label(self.alt_target_var.get())
            if not atid:
                self.status_label.config(text=f"Prompt {pt!r} needs an alt target.")
                return None
            kwargs["alt_target_id"] = atid
        if "action" in sig:
            act = self.action_var.get()
            if not act:
                self.status_label.config(text=f"Prompt {pt!r} needs an action.")
                return None
            kwargs["action"] = act
        if "amount" in sig:
            amt = self.amount_var.get()
            if not amt:
                self.status_label.config(text=f"Prompt {pt!r} needs an amount.")
                return None
            kwargs["amount"] = amt
        if pt == "other":
            kwargs["free_text"] = self.free_text_var.get().strip()
            kwargs["free_choices"] = [v.get().strip() for v in self.free_choice_vars if v.get().strip()]
            if not kwargs["free_text"] or not kwargs["free_choices"]:
                self.status_label.config(text="Free-text prompt needs text + ≥1 choice.")
                return None
        try:
            tool_call, _ctx = build_prompt(self.env, pt, self._current_blob or {}, **kwargs)
            validate_tool_call(tool_call, env=self.env)
        except Exception as e:
            self.status_label.config(text=f"Invalid INTERACT: {e}")
            return None
        return tool_call

    def _build_execution_from_form(self) -> Optional[Dict]:
        tool = self.exec_tool_var.get()
        if not tool:
            self.status_label.config(text="Pick an execution tool.")
            return None
        args: Dict = {}
        if tool != "RELEASE":
            tid = self._id_from_target_label(self.exec_target_var.get())
            if not tid:
                self.status_label.config(text=f"{tool} needs a target.")
                return None
            args["obj"] = tid
        if tool == "POUR":
            amt = self.exec_amount_var.get()
            if not amt:
                self.status_label.config(text="POUR needs an amount.")
                return None
            args["amount"] = amt
        tool_call = {"tool": tool, "args": args}
        try:
            validate_tool_call(tool_call, env=self.env)
        except Exception as e:
            self.status_label.config(text=f"Invalid execution: {e}")
            return None
        return tool_call

    def _refresh_preview(self) -> None:
        """Always render *something* into the preview pane.

        Three sections:
          1. INTERACT preview — full templated text + choices, or a missing-args list.
          2. EXECUTION preview — full tool call, or a missing-args list.
          3. Status line — precondition_status from the factory ("ok" / "warn" / "fail" + message).
        """
        blob = self._current_blob or {}
        lines: List[str] = []

        # ───── INTERACT preview ─────
        pt = self._selected_prompt_type()
        if pt:
            sig = set(prompt_signature(pt))
            kwargs: Dict = {}
            missing: List[str] = []
            if "target" in sig:
                tid = self._id_from_target_label(self.target_var.get())
                if tid:
                    kwargs["target_id"] = tid
                else:
                    missing.append("target")
            if "alt_target" in sig:
                atid = self._id_from_target_label(self.alt_target_var.get())
                if atid:
                    kwargs["alt_target_id"] = atid
                else:
                    missing.append("alt_target")
            if "action" in sig:
                if self.action_var.get():
                    kwargs["action"] = self.action_var.get()
                else:
                    missing.append("action")
            if "amount" in sig:
                if self.amount_var.get():
                    kwargs["amount"] = self.amount_var.get()
                else:
                    missing.append("amount")
            if pt == "other":
                ft = self.free_text_var.get().strip()
                fc = [v.get().strip() for v in self.free_choice_vars if v.get().strip()]
                if ft and fc:
                    kwargs["free_text"] = ft
                    kwargs["free_choices"] = fc
                else:
                    if not ft:
                        missing.append("free_text")
                    if not fc:
                        missing.append("free_choices")

            lines.append(f"── INTERACT [{pt}] ──")
            if missing:
                lines.append(f"⏳ Waiting for: {', '.join(missing)}")
            else:
                try:
                    tc, _ = build_prompt(self.env, pt, blob, **kwargs)
                    lines.append(f"[{tc['args']['kind']}] {tc['args']['text']}")
                    for c in tc["args"]["choices"]:
                        lines.append(f"  · {c}")
                except Exception as e:
                    lines.append(f"⚠ build error: {e}")

            # Precondition status
            level, msg = precondition_status(self.env, pt, blob)
            if level == "fail":
                lines.append(f"❌ ORACLE CHECK: {msg}")
            elif level == "warn":
                lines.append(f"⚠  ORACLE CHECK: {msg}")
            else:
                lines.append(f"✓ ORACLE CHECK: this prompt is consistent with the oracle's tree here.")
        else:
            lines.append("── INTERACT ──")
            lines.append("(pick a Prompt type to preview)")

        # ───── EXECUTION preview ─────
        lines.append("")
        tool = self.exec_tool_var.get()
        if tool:
            args: Dict = {}
            missing: List[str] = []
            if tool != "RELEASE":
                tid = self._id_from_target_label(self.exec_target_var.get())
                if tid:
                    args["obj"] = tid
                else:
                    missing.append("target")
            if tool == "POUR":
                if self.exec_amount_var.get():
                    args["amount"] = self.exec_amount_var.get()
                else:
                    missing.append("amount")
            lines.append(f"── EXECUTE [{tool}] ──")
            if missing:
                lines.append(f"⏳ Waiting for: {', '.join(missing)}")
            else:
                lines.append(f'{{"tool": "{tool}", "args": {args}}}')
        else:
            lines.append("── EXECUTE ──")
            lines.append("(pick a tool to preview)")

        self.preview_text.config(state="normal")
        self.preview_text.delete("1.0", "end")
        self.preview_text.insert("1.0", "\n".join(lines))
        self.preview_text.config(state="disabled")

    # ----------------------------------------------------------- submit ops

    def _on_submit_interact(self) -> None:
        tool_call = self._build_interact_from_form()
        if tool_call is not None:
            self._decision_queue.put(tool_call)

    def _on_submit_execution(self) -> None:
        tool_call = self._build_execution_from_form()
        if tool_call is not None:
            self._decision_queue.put(tool_call)

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
