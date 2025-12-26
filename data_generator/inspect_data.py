from __future__ import annotations

import argparse
import json
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i} in {path}: {e}") from e
    return rows


def _short(s: str, max_len: int) -> str:
    s = " ".join(s.split())
    if max_len <= 0 or len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _fmt_tool_call(tc: Dict[str, Any], max_text: int) -> str:
    tool = tc.get("tool")
    args = tc.get("args", {})
    if tool == "INTERACT":
        kind = args.get("kind", "?")
        text = _short(str(args.get("text", "")), max_text)
        choices = args.get("choices", [])
        if isinstance(choices, list) and choices:
            choices_s = ", ".join(_short(str(c), 40) for c in choices)
            return f"INTERACT[{kind}] {text} | choices: {choices_s}"
        return f"INTERACT[{kind}] {text}"
    if tool in {"APPROACH", "ALIGN_YAW"}:
        return f"{tool}(obj={args.get('obj')})"
    return f"{tool}({args})"


def _episode_groups(rows: Sequence[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    eps: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
    for i, r in enumerate(rows):
        # Preserve file order as a fallback turn index when `t` is not present.
        if isinstance(r, dict) and "_idx" not in r:
            r["_idx"] = i
        eps[int(r["episode_id"])].append(r)
    for ep_id in list(eps.keys()):
        eps[ep_id].sort(key=lambda x: int(x.get("t", x.get("_idx", 0))))
    return dict(eps)


@dataclass
class RepeatFindings:
    max_same_tool_run: int = 1
    max_same_prompt_run: int = 1
    n_duplicate_prompt_transitions: int = 0
    n_duplicate_tool_transitions: int = 0


def _analyze_repeats(ep_rows: Sequence[Dict[str, Any]]) -> RepeatFindings:
    tools: List[str] = []
    prompts: List[Optional[str]] = []
    for r in ep_rows:
        tc = r.get("target_tool_call", {})
        tool = tc.get("tool")
        tools.append(str(tool))
        if tool == "INTERACT":
            prompts.append(str(tc.get("args", {}).get("text", "")))
        else:
            prompts.append(None)

    def max_run(seq: Sequence[Any]) -> int:
        best = 1
        cur = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i - 1] and seq[i] is not None:
                cur += 1
                best = max(best, cur)
            else:
                cur = 1
        return best

    n_dup_tool = sum(1 for i in range(1, len(tools)) if tools[i] == tools[i - 1])
    n_dup_prompt = sum(
        1
        for i in range(1, len(prompts))
        if prompts[i] is not None and prompts[i] == prompts[i - 1]
    )

    return RepeatFindings(
        max_same_tool_run=max_run(tools),
        max_same_prompt_run=max_run(prompts),
        n_duplicate_tool_transitions=n_dup_tool,
        n_duplicate_prompt_transitions=n_dup_prompt,
    )


def _find_yes_no_flipflops(ep_rows: Sequence[Dict[str, Any]]) -> int:
    """
    Counts how many times the user answers YES then NO (or NO then YES) to the same prompt text
    within an episode.
    """
    flips = 0
    # Walk assistant prompts and user replies from memory.past_dialogs snapshots.
    # We use the snapshot at time t, which includes the dialog history up to that point.
    for r in ep_rows:
        past = (r.get("memory") or {}).get("past_dialogs") or []
        # We only need the last two turns to detect flips, but we also want
        # to avoid counting the same flip many times across timesteps.
        if len(past) < 4:
            continue
        a1, u1, a2, u2 = past[-4], past[-3], past[-2], past[-1]
        if a1.get("role") != "assistant" or u1.get("role") != "user":
            continue
        if a2.get("role") != "assistant" or u2.get("role") != "user":
            continue
        if str(a1.get("content", "")).strip() != str(a2.get("content", "")).strip():
            continue
        resp1 = str(u1.get("content", "")).strip().upper()
        resp2 = str(u2.get("content", "")).strip().upper()
        if {resp1, resp2} == {"YES", "NO"}:
            flips += 1
    return flips


def _print_episode(
    ep_id: int,
    ep_rows: Sequence[Dict[str, Any]],
    *,
    max_t: Optional[int],
    show_objects: bool,
    show_gripper: bool,
    show_memory: bool,
    max_text: int,
    wrap: int,
) -> None:
    findings = _analyze_repeats(ep_rows)
    flipflops = _find_yes_no_flipflops(ep_rows)

    print(f"\n=== EPISODE {ep_id} | steps={len(ep_rows)} ===")
    print(
        "repeats:",
        f"max_same_tool_run={findings.max_same_tool_run},",
        f"max_same_prompt_run={findings.max_same_prompt_run},",
        f"dup_tool_transitions={findings.n_duplicate_tool_transitions},",
        f"dup_prompt_transitions={findings.n_duplicate_prompt_transitions},",
        f"yes/no_flipflops={flipflops}",
    )

    tool_counts = Counter(r.get("target_tool_call", {}).get("tool") for r in ep_rows)
    print("tool_counts:", dict(tool_counts))

    for r in ep_rows:
        t = int(r.get("t", r.get("_idx", 0)))
        if max_t is not None and t > max_t:
            break

        tc = r.get("target_tool_call", {})
        line = f"turn={t:02d}  {_fmt_tool_call(tc, max_text=max_text)}"
        print(line)

        if show_gripper:
            gh = r.get("gripper_hist") or []
            if gh:
                g = gh[-1]
                print(f"      gripper: cell={g.get('cell')} yaw={g.get('yaw')} z={g.get('z')}")

        if show_objects:
            objs = r.get("objects") or []
            # Sort by id for stable display.
            objs = sorted(objs, key=lambda o: str(o.get("id")))
            concise = [f"{o.get('id')}:{o.get('label')}@{o.get('cell')}/{o.get('yaw')}" for o in objs]
            print("      objects:", ", ".join(concise))

        mem = r.get("memory") or {}
        if show_memory:
            cands = mem.get("candidates", [])
            last_calls = mem.get("last_tool_calls", [])
            print(f"      memory: n_interactions={mem.get('n_interactions')} candidates={cands} last={last_calls}")
            past = mem.get("past_dialogs", [])
            if past:
                a = past[-1]
                role = a.get("role")
                content = _short(str(a.get("content", "")), max_text)
                wrapped = textwrap.fill(
                    f"{role}: {content}",
                    width=wrap,
                    subsequent_indent="      ",
                )
                print("      last_dialog:", wrapped)


def _summary(rows: Sequence[Dict[str, Any]]) -> None:
    tool_counts = Counter(r.get("target_tool_call", {}).get("tool") for r in rows)
    interact_kind_counts = Counter(
        (r.get("target_tool_call", {}).get("args", {}) or {}).get("kind")
        for r in rows
        if (r.get("target_tool_call", {}) or {}).get("tool") == "INTERACT"
    )
    episodes = _episode_groups(rows)
    inter_counts = {ep: int((eps[0].get("memory") or {}).get("n_interactions", 0)) for ep, eps in episodes.items() if eps}

    print(f"rows={len(rows)} episodes={len(episodes)}")
    print("tool_distribution:", dict(tool_counts))
    if interact_kind_counts:
        # Filter None for cleaner output.
        interact_kind_counts.pop(None, None)
        print("interact_kind_distribution:", dict(interact_kind_counts))

    if episodes:
        lens = [len(v) for v in episodes.values()]
        print(f"episode_len: min={min(lens)} max={max(lens)} avg={sum(lens)/len(lens):.2f}")

        # Object label coverage + overlap stats (computed per-episode from the first row).
        label_counts: Counter[str] = Counter()
        episodes_with_overlap = 0
        max_overlap_per_ep: List[int] = []
        for ep_rows in episodes.values():
            if not ep_rows:
                continue
            objs = ep_rows[0].get("objects") or []
            for o in objs:
                label_counts[str(o.get("label"))] += 1
            cell_counts = Counter(str(o.get("cell")) for o in objs)
            max_mult = max(cell_counts.values()) if cell_counts else 1
            max_overlap_per_ep.append(max_mult)
            if any(v >= 2 for v in cell_counts.values()):
                episodes_with_overlap += 1

        if label_counts:
            uniq = len(label_counts)
            most_common = label_counts.most_common(10)
            print(f"object_label_coverage: unique={uniq} (showing top10 by episode-count)")
            print("object_label_counts:", dict(most_common))

        if max_overlap_per_ep:
            overlap_rate = episodes_with_overlap / max(1, len(episodes))
            print(
                "object_cell_overlap:",
                f"episodes_with_overlap={episodes_with_overlap}/{len(episodes)} ({overlap_rate:.2%}),",
                f"max_objects_sharing_cell: min={min(max_overlap_per_ep)} max={max(max_overlap_per_ep)} avg={sum(max_overlap_per_ep)/len(max_overlap_per_ep):.2f}",
            )

        # Gripper symbol coverage (cells/yaws/z over all 6-history entries).
        gripper_cells: Counter[str] = Counter()
        gripper_yaws: Counter[str] = Counter()
        gripper_z: Counter[str] = Counter()
        cand_sizes: Counter[int] = Counter()
        for r in rows:
            gh = r.get("gripper_hist") or []
            for g in gh:
                if "cell" in g:
                    gripper_cells[str(g.get("cell"))] += 1
                if "yaw" in g:
                    gripper_yaws[str(g.get("yaw"))] += 1
                if "z" in g:
                    gripper_z[str(g.get("z"))] += 1
            mem = r.get("memory") or {}
            cands = mem.get("candidates") or []
            if isinstance(cands, list):
                cand_sizes[len(cands)] += 1

        if gripper_cells:
            print(f"gripper_cells: unique={len(gripper_cells)} top5={dict(gripper_cells.most_common(5))}")
        if gripper_yaws:
            print(f"gripper_yaws: unique={len(gripper_yaws)} top5={dict(gripper_yaws.most_common(5))}")
        if gripper_z:
            print(f"gripper_z: unique={len(gripper_z)} counts={dict(gripper_z)}")
        if cand_sizes:
            print(f"candidate_set_size: unique={len(cand_sizes)} top5={dict(cand_sizes.most_common(5))}")

        rep = {ep: _analyze_repeats(eps) for ep, eps in episodes.items()}
        worst_prompt = max(rep.items(), key=lambda kv: kv[1].max_same_prompt_run)
        worst_tool = max(rep.items(), key=lambda kv: kv[1].max_same_tool_run)
        print(
            "worst_prompt_run:",
            f"episode={worst_prompt[0]} run={worst_prompt[1].max_same_prompt_run}",
        )
        print(
            "worst_tool_run:",
            f"episode={worst_tool[0]} run={worst_tool[1].max_same_tool_run}",
        )

    if inter_counts:
        print("n_interactions (first-snapshot per ep):", dict(sorted(inter_counts.items())))


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Inspect and summarize data_generator JSONL datasets.")
    ap.add_argument("--path", type=str, required=True, help="Path to a .jsonl file.")
    ap.add_argument("--episode", type=int, action="append", default=None, help="Episode id(s) to print.")
    ap.add_argument(
        "--episode-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="Inclusive episode id range to print (e.g. --episode-range 4 12).",
    )
    ap.add_argument(
        "--episode-step",
        type=int,
        default=1,
        help="Step for --episode-range (default: 1).",
    )
    ap.add_argument("--max-t", type=int, default=None, help="Max timestep to print per episode.")
    ap.add_argument("--summary", action="store_true", help="Print dataset summary.")
    ap.add_argument("--show-objects", action="store_true", help="Print objects each step.")
    ap.add_argument("--show-gripper", action="store_true", help="Print current gripper pose each step.")
    ap.add_argument("--show-memory", action="store_true", help="Print memory snapshot and last dialog each step.")
    ap.add_argument("--max-text", type=int, default=140, help="Max chars for prompt/dialog text.")
    ap.add_argument("--wrap", type=int, default=120, help="Wrap width for long lines.")
    args = ap.parse_args(argv)

    path = Path(args.path)
    rows = _load_jsonl(path)
    episodes = _episode_groups(rows)

    if args.summary:
        _summary(rows)

    # Default: print a couple episodes if none are specified.
    ep_ids: List[int]
    if args.episode:
        ep_ids = list(args.episode)
    elif args.episode_range:
        start, end = args.episode_range
        step = int(args.episode_step)
        if step <= 0:
            raise ValueError("--episode-step must be >= 1")
        if start > end:
            start, end = end, start
        ep_ids = list(range(start, end + 1, step))
    else:
        ep_ids = sorted(episodes.keys())[:3]

    for ep_id in ep_ids:
        ep_rows = episodes.get(ep_id, [])
        if not ep_rows:
            print(f"\n=== EPISODE {ep_id} ===\n(no rows found)")
            continue
        _print_episode(
            ep_id,
            ep_rows,
            max_t=args.max_t,
            show_objects=args.show_objects,
            show_gripper=args.show_gripper,
            show_memory=args.show_memory,
            max_text=args.max_text,
            wrap=args.wrap,
        )


if __name__ == "__main__":
    main()



