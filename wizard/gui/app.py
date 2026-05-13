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
COL_SUCCESS     = "#0f766e"
COL_DANGER      = "#b42318"
COL_ASK_BG      = "#e8f1ff"
COL_ACT_BG      = "#ecfdf3"

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
FONT_PREVIEW   = ("Courier", 9)


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
        self.root.configure(bg=COL_BG)
        self.root.geometry("1320x960")

        # ─── Header (dark strip) ──────────────────────────────────────
        self.header = tk.Frame(self.root, bg=COL_HEADER_BG, padx=14, pady=8)
        self.header.pack(fill="x")
        self.header_label = tk.Label(
            self.header, text="Waiting for first alert…",
            fg=COL_HEADER_FG, bg=COL_HEADER_BG, font=FONT_HEADER,
        )
        self.header_label.pack(side="left")

        # Full-width alert strip below header — visually distinct & wide
        self.alert_strip = tk.Frame(self.root, bg=COL_BG, padx=14, pady=6)
        self.alert_strip.pack(fill="x")
        self.alert_banner = tk.Label(
            self.alert_strip, text="", bg=COL_BG, fg=COL_HEADER_BG,
            font=FONT_SECTION, anchor="w", padx=12, pady=6,
        )
        self.alert_banner.pack(fill="x")

        # ─── Mid section: grid (left) + dialog + scene (right stacked) ─
        mid = tk.Frame(self.root, bg=COL_BG)
        mid.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        # Left: grid view
        left = tk.Frame(mid, bg=COL_PANEL, bd=1, relief="solid",
                        highlightbackground=COL_BORDER, highlightthickness=1)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        tk.Label(left, text="🗺  WORKSPACE  (top-down · row A is far)",
                 bg=COL_PANEL, fg=COL_HEADER_BG, font=FONT_SECTION,
                 anchor="w", padx=8, pady=6).pack(fill="x")
        self.grid_view = GridView()
        self.canvas = FigureCanvasTkAgg(self.grid_view.fig, master=left)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=(0, 6))

        # Right: dialog (top) + scene (bottom)
        right = tk.Frame(mid, bg=COL_BG, width=520)
        right.pack(side="right", fill="y", padx=(6, 0))
        right.pack_propagate(False)

        self._build_dialog_panel(right)
        self._build_scene_panel(right)

        self._build_decision_panel()

    def _build_decision_panel(self) -> None:
        decision = tk.Frame(self.root, padx=10, pady=8, bg=COL_BG)
        decision.pack(fill="x")

        title_row = tk.Frame(decision, bg=COL_BG)
        title_row.pack(fill="x", pady=(0, 4))
        tk.Label(
            title_row, text="Decision", bg=COL_BG, fg=COL_HEADER_BG,
            font=FONT_SECTION,
        ).pack(side="left")
        tk.Label(
            title_row,
            text="Ask and act are separate turns; an ask waits for the user's reply before any state-changing tool.",
            bg=COL_BG, fg=COL_MUTED, font=FONT_SMALL,
        ).pack(side="right")

        body = tk.Frame(decision, bg=COL_BG)
        body.pack(fill="x")

        self.decision_notebook = ttk.Notebook(body)
        self.decision_notebook.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.decision_notebook.bind("<<NotebookTabChanged>>", lambda _e: self._on_form_changed())

        interact_frame = tk.Frame(self.decision_notebook, bg=COL_PANEL, padx=10, pady=10)
        execute_frame = tk.Frame(self.decision_notebook, bg=COL_PANEL, padx=10, pady=10)
        self.decision_notebook.add(interact_frame, text="Ask user")
        self.decision_notebook.add(execute_frame, text="Take action")

        preview_frame = tk.Frame(
            body, bg=COL_PANEL, bd=1, relief="solid",
            highlightbackground=COL_BORDER, highlightthickness=1,
        )
        preview_frame.pack(side="right", fill="both")
        tk.Label(
            preview_frame, text="Oracle-format preview", bg=COL_PANEL,
            fg=COL_HEADER_BG, font=FONT_LABEL, anchor="w", padx=8, pady=6,
        ).pack(fill="x")
        self.preview_text = tk.Text(
            preview_frame, height=15, width=58, font=FONT_PREVIEW,
            bg="#ffffff", fg=COL_VALUE, state="disabled", wrap="word",
            padx=8, pady=8, relief="flat",
        )
        self.preview_text.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        tk.Label(
            interact_frame, text="Oracle template", bg=COL_PANEL,
            fg=COL_LABEL, font=FONT_LABEL,
        ).pack(anchor="w")
        self.prompt_type_var = tk.StringVar()
        self._prompt_type_keys: List[str] = list(list_prompt_types(self.env))
        self._prompt_type_labels: List[str] = [
            display_label(pt) for pt in self._prompt_type_keys
        ]
        self.prompt_type_combo = ttk.Combobox(
            interact_frame, textvariable=self.prompt_type_var, state="readonly",
            values=self._prompt_type_labels, width=82,
        )
        self.prompt_type_combo.pack(fill="x", pady=(2, 4))
        self.prompt_type_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_form_changed())
        self.prompt_hint_label = tk.Label(
            interact_frame, text="Pick the oracle prompt shape first.",
            bg=COL_ASK_BG, fg=COL_HEADER_BG, font=FONT_SMALL,
            anchor="w", justify="left", padx=8, pady=5,
        )
        self.prompt_hint_label.pack(fill="x", pady=(0, 8))

        self.target_var = tk.StringVar()
        self.target_row = self._build_combo_row(
            interact_frame, "Target object",
            "Only shown for templates whose oracle context needs a primary object.",
            self.target_var, width=54,
        )
        self.target_combo = self.target_row.combo
        self.target_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_form_changed())

        self.alt_target_var = tk.StringVar()
        self.alt_target_row = self._build_combo_row(
            interact_frame, "Alternative target",
            "Used by redirect templates such as covered-cube or full-cup suggestions.",
            self.alt_target_var, width=54,
        )
        self.alt_target_combo = self.alt_target_row.combo
        self.alt_target_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_form_changed())

        self.action_var = tk.StringVar()
        self.action_row = self._build_combo_row(
            interact_frame, "Action to confirm",
            "Used only by the generic confirm template.",
            self.action_var, width=18,
        )
        self.action_combo = self.action_row.combo
        self.action_combo["values"] = list(valid_actions(self.env, "confirm"))
        self.action_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_form_changed())

        self.amount_var = tk.StringVar()
        self.amount_row = self._build_combo_row(
            interact_frame, "Pour amount",
            "Used only by amount-confirmation prompts.",
            self.amount_var, width=18,
        )
        self.amount_combo = self.amount_row.combo
        self.amount_combo["values"] = list(valid_amounts())
        self.amount_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_form_changed())

        self.free_text_row = tk.Frame(interact_frame, bg=COL_PANEL)
        tk.Label(
            self.free_text_row, text="Free-text fallback", bg=COL_PANEL,
            fg=COL_LABEL, font=FONT_LABEL,
        ).pack(anchor="w")
        self.free_text_var = tk.StringVar()
        self.free_text_entry = ttk.Entry(
            self.free_text_row, textvariable=self.free_text_var,
            width=58, state="disabled",
        )
        self.free_text_entry.pack(fill="x", pady=(2, 4))
        self.free_text_entry.bind("<KeyRelease>", lambda _e: self._on_form_changed())

        self.free_choice_row = tk.Frame(interact_frame, bg=COL_PANEL)
        tk.Label(
            self.free_choice_row, text="Choices", bg=COL_PANEL,
            fg=COL_LABEL, font=FONT_LABEL,
        ).pack(anchor="w")
        self.free_choice_vars: List[tk.StringVar] = []
        self.free_choice_entries: List[ttk.Entry] = []
        for i in range(MAX_INTERACT_CHOICES):
            opt = tk.Frame(self.free_choice_row, bg=COL_PANEL)
            opt.pack(fill="x", pady=1)
            tk.Label(
                opt, text=f"{i + 1})", bg=COL_PANEL, fg=COL_MUTED,
                font=FONT_SMALL, width=3,
            ).pack(side="left")
            v = tk.StringVar()
            self.free_choice_vars.append(v)
            ent = ttk.Entry(opt, textvariable=v, width=58, state="disabled")
            ent.pack(side="left", fill="x", expand=True)
            ent.bind("<KeyRelease>", lambda _e: self._on_form_changed())
            self.free_choice_entries.append(ent)

        tk.Label(
            execute_frame, text="Execution tool", bg=COL_PANEL,
            fg=COL_LABEL, font=FONT_LABEL,
        ).pack(anchor="w")
        self.exec_tool_var = tk.StringVar()
        self.exec_tool_combo = ttk.Combobox(
            execute_frame, textvariable=self.exec_tool_var,
            state="readonly", width=18,
        )
        self.exec_tool_combo.pack(anchor="w", pady=(2, 8))
        self.exec_tool_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_form_changed())

        self.exec_target_var = tk.StringVar()
        self.exec_target_row = self._build_combo_row(
            execute_frame, "Object",
            "Filtered to the objects that the selected tool can legally use.",
            self.exec_target_var, width=46,
        )
        self.exec_target_combo = self.exec_target_row.combo
        self.exec_target_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_form_changed())

        self.exec_amount_var = tk.StringVar()
        self.exec_amount_row = self._build_combo_row(
            execute_frame, "Amount",
            "Required only for POUR.",
            self.exec_amount_var, width=18,
        )
        self.exec_amount_combo = self.exec_amount_row.combo
        self.exec_amount_combo["values"] = list(valid_amounts())
        self.exec_amount_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_form_changed())

        submit_row = tk.Frame(self.root, pady=6, bg=COL_BG)
        submit_row.pack(fill="x")
        self.submit_button = ttk.Button(
            submit_row, text="Submit ask", command=self._on_submit_current,
        )
        self.submit_button.pack(side="left", padx=6)
        ttk.Button(
            submit_row, text="Defer (skip this alert)", command=self._on_skip,
        ).pack(side="left", padx=6)
        self.status_label = tk.Label(
            submit_row, text="", bg=COL_BG, fg=COL_DANGER, font=FONT_LABEL,
        )
        self.status_label.pack(side="right", padx=8)
        self._on_form_changed()

    def _build_combo_row(
        self,
        parent: tk.Widget,
        label: str,
        help_text: str,
        variable: tk.StringVar,
        *,
        width: int,
    ) -> tk.Frame:
        row = tk.Frame(parent, bg=COL_PANEL)
        row.pack(fill="x", pady=(0, 8))
        tk.Label(
            row, text=label, bg=COL_PANEL, fg=COL_LABEL,
            font=FONT_LABEL,
        ).pack(anchor="w")
        control = tk.Frame(row, bg=COL_PANEL)
        control.pack(fill="x", pady=(2, 2))
        combo = ttk.Combobox(
            control, textvariable=variable, state="readonly", width=width,
        )
        combo.pack(side="left", fill="x", expand=True)
        tk.Label(
            row, text=help_text, bg=COL_PANEL, fg=COL_MUTED,
            font=FONT_SMALL, anchor="w", justify="left",
        ).pack(fill="x")
        row.control = control  # type: ignore[attr-defined]
        row.combo = combo  # type: ignore[attr-defined]
        return row

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
        rkey = reason.value if reason else ""
        label = ALERT_LABEL.get(rkey, f"⚠  ALERT · {rkey}") if rkey else "⚠  ALERT"
        self.alert_banner.config(
            text=f"{label}    ·    please decide",
            bg=COL_ALERT_BG, fg="#1a1a1a",
        )

        # Render grid.
        self.grid_view.render(blob)
        self.canvas.draw()

        # Render the new dialog + scene panels.
        self._render_dialog_history(blob.get("memory", {}).get("past_dialogs") or [])
        self._render_scene_state(blob)

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
        self.decision_notebook.select(0)
        self._on_form_changed()  # refresh dropdown contents + preview
        self.status_label.config(text="")

    # ─────────────────────────────── dialog + scene panels ──────────

    def _build_dialog_panel(self, parent: tk.Widget) -> None:
        """Scrollable card list of past INTERACT turns with bold user replies."""
        title_bar = tk.Frame(parent, bg=COL_BG)
        title_bar.pack(fill="x", padx=2, pady=(2, 0))
        tk.Label(title_bar, text="💬  DIALOG HISTORY  (user replies)",
                 bg=COL_BG, fg=COL_HEADER_BG, font=FONT_SECTION,
                 anchor="w").pack(side="left")
        self.dialog_count_label = tk.Label(title_bar, text="",
                                           bg=COL_BG, fg=COL_MUTED,
                                           font=FONT_SMALL)
        self.dialog_count_label.pack(side="right")

        outer = tk.Frame(parent, bg=COL_PANEL, bd=1, relief="solid",
                         highlightbackground=COL_BORDER, highlightthickness=1)
        outer.pack(fill="both", expand=True, padx=2, pady=(4, 6))

        canvas = tk.Canvas(outer, bg=COL_PANEL, highlightthickness=0, height=340)
        vsb = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.dialog_inner = tk.Frame(canvas, bg=COL_PANEL)
        self._dialog_window = canvas.create_window(
            (0, 0), window=self.dialog_inner, anchor="nw",
        )

        def _on_inner_configure(_e=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(e):
            canvas.itemconfig(self._dialog_window, width=e.width)

        self.dialog_inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_wheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", _on_wheel))
        canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))

        self._dialog_canvas = canvas

    def _build_scene_panel(self, parent: tk.Widget) -> None:
        """Compact structured key/value rows for the rest of the symbolic state."""
        tk.Label(parent, text="📋  SCENE STATE", bg=COL_BG,
                 fg=COL_HEADER_BG, font=FONT_SECTION, anchor="w",
                 ).pack(fill="x", padx=2, pady=(6, 2))

        card = tk.Frame(parent, bg=COL_PANEL, bd=1, relief="solid",
                        highlightbackground=COL_BORDER, highlightthickness=1)
        card.pack(fill="x", padx=2, pady=(0, 4))

        inner = tk.Frame(card, bg=COL_PANEL, padx=10, pady=8)
        inner.pack(fill="x")

        self._scene_value_labels: Dict[str, tk.Label] = {}
        rows = [
            ("mode",        "User mode"),
            ("candidates",  "Candidates"),
            ("excluded",    "Excluded"),
            ("last_action", "Last action"),
            ("last_tools",  "Last 3 tools"),
            ("gripper",     "Gripper trail"),
            ("last_prompt", "Last prompt"),
        ]
        for i, (key, label) in enumerate(rows):
            tk.Label(inner, text=label, bg=COL_PANEL, fg=COL_LABEL,
                     font=FONT_LABEL, anchor="nw").grid(
                row=i, column=0, sticky="nw", padx=(0, 10), pady=2)
            v = tk.Label(inner, text="—", bg=COL_PANEL, fg=COL_VALUE,
                         font=FONT_BASE, anchor="w", justify="left",
                         wraplength=360)
            v.grid(row=i, column=1, sticky="we", pady=2)
            self._scene_value_labels[key] = v
        inner.columnconfigure(1, weight=1)

    def _make_dialog_card(self, parent: tk.Widget, turn_idx: int,
                          dialog: Dict, is_latest: bool) -> None:
        kind = (dialog.get("kind") or "").upper()
        chip_bg, chip_fg = KIND_CHIP.get(kind, ("#eceff1", "#37474f"))
        reply = dialog.get("reply") or "(no reply yet)"
        bg, fg = _classify_reply(reply)

        card = tk.Frame(
            parent, bg=COL_PANEL,
            highlightthickness=3 if is_latest else 1,
            highlightbackground=COL_LATEST if is_latest else COL_BORDER,
            bd=0,
        )
        card.pack(fill="x", padx=8, pady=(10 if is_latest else 4, 4))

        inner = tk.Frame(card, bg=COL_PANEL, padx=10, pady=8)
        inner.pack(fill="x")

        # Header strip
        head = tk.Frame(inner, bg=COL_PANEL)
        head.pack(fill="x")
        tk.Label(head, text=f"Turn {turn_idx}", bg=COL_PANEL, fg=COL_MUTED,
                 font=FONT_SMALL).pack(side="left")
        tk.Label(head, text=f"  {kind}  ", bg=chip_bg, fg=chip_fg,
                 font=FONT_CHIP, padx=6, pady=1).pack(side="left", padx=(8, 0))
        if is_latest:
            tk.Label(head, text="  ★ LATEST  ", bg=COL_LATEST, fg="#1a1a1a",
                     font=FONT_CHIP, padx=6, pady=1).pack(side="right")

        # Prompt
        prompt = dialog.get("prompt") or "(no prompt)"
        tk.Label(inner, text=prompt, bg=COL_PANEL, fg=COL_VALUE,
                 font=FONT_BASE, wraplength=440, justify="left",
                 anchor="w").pack(fill="x", pady=(6, 2))

        # Choices
        choices = dialog.get("choices") or []
        if choices:
            choices_str = "Choices:   " + "    ·   ".join(str(c) for c in choices)
            tk.Label(inner, text=choices_str, bg=COL_PANEL, fg=COL_MUTED,
                     font=FONT_MUTED, wraplength=440, justify="left",
                     anchor="w").pack(fill="x", pady=(0, 8))

        # Reply badge — the headline of the card
        reply_row = tk.Frame(inner, bg=COL_PANEL)
        reply_row.pack(fill="x")
        tk.Label(reply_row, text="USER REPLY  →", bg=COL_PANEL,
                 fg=COL_LABEL, font=FONT_LABEL).pack(side="left")
        font = FONT_REPLY_HI if is_latest else FONT_REPLY
        tk.Label(reply_row, text=f"  {reply}  ", bg=bg, fg=fg, font=font,
                 padx=12, pady=4).pack(side="left", padx=8)

    def _render_dialog_history(self, past_dialogs: List[Dict]) -> None:
        for child in self.dialog_inner.winfo_children():
            child.destroy()

        n = len(past_dialogs)
        self.dialog_count_label.config(
            text=f"{n} turn(s)" if n else "no turns yet"
        )

        if not past_dialogs:
            tk.Label(
                self.dialog_inner,
                text="(no past dialogs — this is the first interaction)",
                bg=COL_PANEL, fg=COL_MUTED, font=FONT_MUTED, pady=24,
            ).pack(fill="x")
            return

        # Most-recent first so the latest reply sits at the top.
        for idx, dialog in enumerate(reversed(past_dialogs)):
            turn_idx = n - idx
            self._make_dialog_card(self.dialog_inner, turn_idx, dialog,
                                   is_latest=(idx == 0))

        self.dialog_inner.update_idletasks()
        self._dialog_canvas.yview_moveto(0.0)

    def _render_scene_state(self, blob: Dict) -> None:
        mem = blob.get("memory") or {}
        user_state = blob.get("user_state") or {}
        hist = blob.get("gripper_hist") or []
        L = self._scene_value_labels

        L["mode"].config(text=user_state.get("mode", "—"), fg=COL_VALUE)

        cands = mem.get("candidates") or []
        L["candidates"].config(text=(", ".join(cands) if cands else "—"))

        excl = mem.get("excluded_obj_ids") or []
        L["excluded"].config(text=(", ".join(excl) if excl else "—"))

        la = mem.get("last_action") or {}
        if la:
            outcome = (la.get("outcome") or "?").lower()
            fg = (COL_SUCCESS if outcome == "success"
                  else COL_DANGER if outcome in ("fail", "failure", "timeout")
                  else COL_VALUE)
            tool = la.get("tool", "?")
            obj = la.get("obj", "")
            L["last_action"].config(
                text=f"{tool}({obj}) → {outcome}", fg=fg)
        else:
            L["last_action"].config(text="—", fg=COL_VALUE)

        tools = mem.get("last_tool_calls") or []
        L["last_tools"].config(text=(" → ".join(tools) if tools else "—"))

        if hist:
            trail = " → ".join(g.get("cell", "?") for g in hist)
            cur = hist[-1]
            L["gripper"].config(
                text=f"{trail}      yaw={cur.get('yaw','?')}   z={cur.get('z','?')}"
            )
        else:
            L["gripper"].config(text="—")

        lp = mem.get("last_prompt") or {}
        if lp:
            lp_text = lp.get("text") or lp.get("prompt") or ""
            L["last_prompt"].config(
                text=f"[{lp.get('kind','?')}]  {lp_text}")
        else:
            L["last_prompt"].config(text="—")

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
        mem = (self._current_blob or {}).get("memory") or {}
        candidates = set(mem.get("candidates") or [])
        excluded = set(mem.get("excluded_obj_ids") or [])
        extras = []
        if obj.get("id") in candidates: extras.append("candidate")
        if obj.get("id") in excluded: extras.append("excluded")
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
        pt = self._selected_prompt_type() or ""

        # Interaction target list filtered by prompt type.
        sig = set(prompt_signature(pt))
        if pt and objs:
            int_targets = valid_targets(self.env, pt, blob)
            target_values = [self._target_label(o) for o in int_targets]
            self.target_combo["values"] = target_values
            if self.target_var.get() and self.target_var.get() not in target_values:
                self.target_var.set("")
            # Alt target gets the same pool, minus the chosen primary.
            primary_id = self._id_from_target_label(self.target_var.get())
            alt_pool = [o for o in int_targets if o["id"] != primary_id]
            alt_values = [self._target_label(o) for o in alt_pool]
            self.alt_target_combo["values"] = alt_values
            if self.alt_target_var.get() and self.alt_target_var.get() not in alt_values:
                self.alt_target_var.set("")

        uses = []
        if "target" in sig:
            uses.append("target")
        if "alt_target" in sig:
            uses.append("alternative target")
        if "action" in sig:
            uses.append("action")
        if "amount" in sig:
            uses.append("amount")
        if pt == "other":
            uses.append("free text")
        if pt:
            detail = ", ".join(uses) if uses else "no extra fields"
            self.prompt_hint_label.config(text=f"{pt}: oracle template uses {detail}.")
        else:
            self.prompt_hint_label.config(text="Pick the oracle prompt shape first.")

        self._show_row(self.target_row, "target" in sig)
        self._show_row(self.alt_target_row, "alt_target" in sig)
        self._show_row(self.action_row, "action" in sig)
        self._show_row(self.amount_row, "amount" in sig)
        self._show_row(self.free_text_row, pt == "other")
        self._show_row(self.free_choice_row, pt == "other")
        self._set_widget_state(self.action_combo, "action" in sig)
        self._set_widget_state(self.amount_combo, "amount" in sig)
        self._set_widget_state(self.target_combo, "target" in sig)
        self._set_widget_state(self.alt_target_combo, "alt_target" in sig)
        self._set_widget_state(self.free_text_entry, pt == "other")
        for ent in self.free_choice_entries:
            self._set_widget_state(ent, pt == "other")

        # Execution target list filtered by tool.
        tool = self.exec_tool_var.get()
        if tool and objs:
            pool = self._exec_targets_for_tool(blob, tool)
            exec_values = [self._target_label(o) for o in pool]
            self.exec_target_combo["values"] = exec_values
            if self.exec_target_var.get() and self.exec_target_var.get() not in exec_values:
                self.exec_target_var.set("")
        self._show_row(self.exec_target_row, bool(tool and tool != "RELEASE"))
        self._show_row(self.exec_amount_row, tool == "POUR")
        self._set_widget_state(self.exec_target_combo, bool(tool and tool != "RELEASE"))
        self._set_widget_state(self.exec_amount_combo, tool == "POUR")

        if hasattr(self, "submit_button"):
            self.submit_button.config(
                text="Submit ask" if self._active_decision_mode() == "ask" else "Submit action"
            )

        # Live preview.
        self._refresh_preview()

    def _active_decision_mode(self) -> str:
        try:
            return "act" if self.decision_notebook.index("current") == 1 else "ask"
        except Exception:
            return "ask"

    @staticmethod
    def _show_row(row: tk.Widget, visible: bool) -> None:
        if visible:
            row.pack(fill="x", pady=(0, 8))
        else:
            row.pack_forget()

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
        """Render the active decision lane in the same shape the oracle emits."""
        blob = self._current_blob or {}
        lines: List[str] = []
        mode = self._active_decision_mode()

        if mode == "ask":
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
                        missing.append("alternative target")
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
                            missing.append("free text")
                        if not fc:
                            missing.append("choices")

                lines.append(f"INTERACT / {pt}")
                lines.append("")
                if missing:
                    lines.append(f"Waiting for: {', '.join(missing)}")
                else:
                    try:
                        tc, _ = build_prompt(self.env, pt, blob, **kwargs)
                        lines.append(f"kind: {tc['args']['kind']}")
                        lines.append(f"text: {tc['args']['text']}")
                        lines.append("choices:")
                        for c in tc["args"]["choices"]:
                            lines.append(f"  {c}")
                    except Exception as e:
                        lines.append(f"Build error: {e}")

                level, msg = precondition_status(self.env, pt, blob)
                lines.append("")
                if level == "fail":
                    lines.append(f"Oracle check: FAIL - {msg}")
                elif level == "warn":
                    lines.append(f"Oracle check: WARN - {msg}")
                else:
                    lines.append("Oracle check: OK - consistent with the oracle tree here.")
                lines.append("")
                lines.append("Flow: submit only this INTERACT turn. The runner records the user reply before the next decision.")
            else:
                lines.append("INTERACT")
                lines.append("")
                lines.append("Pick an oracle template to preview the exact prompt and choices.")
            bg = COL_ASK_BG
        else:
            tool = self.exec_tool_var.get()
            if tool:
                args: Dict = {}
                missing: List[str] = []
                if tool != "RELEASE":
                    tid = self._id_from_target_label(self.exec_target_var.get())
                    if tid:
                        args["obj"] = tid
                    else:
                        missing.append("object")
                if tool == "POUR":
                    if self.exec_amount_var.get():
                        args["amount"] = self.exec_amount_var.get()
                    else:
                        missing.append("amount")
                lines.append(f"EXECUTE / {tool}")
                lines.append("")
                if missing:
                    lines.append(f"Waiting for: {', '.join(missing)}")
                else:
                    lines.append(f"tool: {tool}")
                    lines.append(f"args: {args}")
                lines.append("")
                lines.append("Flow: this changes the schematic state immediately and logs one execution tool call.")
            else:
                lines.append("EXECUTE")
                lines.append("")
                lines.append("Pick a tool to preview the validated action call.")
            bg = COL_ACT_BG

        self.preview_text.config(state="normal")
        self.preview_text.config(bg=bg)
        self.preview_text.delete("1.0", "end")
        self.preview_text.insert("1.0", "\n".join(lines))
        self.preview_text.config(state="disabled")

    # ----------------------------------------------------------- submit ops

    def _on_submit_current(self) -> None:
        if self._active_decision_mode() == "act":
            self._on_submit_execution()
        else:
            self._on_submit_interact()

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
