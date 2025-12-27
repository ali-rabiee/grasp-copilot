from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from data_generator.oracle import validate_tool_call

from .utils import json_loads_strict, set_seed


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _extract_first_json_object(s: str) -> Optional[str]:
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


def _parse_model_json(s: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Returns (obj, error). Attempts strict JSON parse, then substring extraction for "JSON + extra".
    """
    try:
        obj = json_loads_strict(s)
        if isinstance(obj, dict):
            return obj, None
        return None, "not_an_object"
    except Exception as e:
        extracted = _extract_first_json_object(s)
        if extracted:
            try:
                obj = json_loads_strict(extracted)
                if isinstance(obj, dict):
                    return obj, None
                return None, "not_an_object"
            except Exception as e2:
                return None, f"json_error_after_extract:{e2}"
        return None, f"json_error:{e}"


def _normalize_tool_call(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize common model drift into the oracle schema:
      - tool name casing
      - INTERACT.kind casing
      - drop extra keys
    """
    tool = obj.get("tool")
    args = obj.get("args")
    if isinstance(tool, str):
        tool = tool.strip().upper()
    if tool == "INTERACT" and isinstance(args, dict):
        kind = args.get("kind", args.get("type"))
        if isinstance(kind, str):
            kind = kind.strip().upper()
        if kind not in {"QUESTION", "SUGGESTION", "CONFIRM"}:
            kind = "QUESTION"
        text = args.get("text", "")
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        choices = args.get("choices", [])
        if not isinstance(choices, list):
            choices = []
        choices = [str(c) for c in choices][:10]  # keep some for diagnostics; validator enforces <=5
        return {"tool": "INTERACT", "args": {"kind": kind, "text": text, "choices": choices}}
    if tool in {"APPROACH", "ALIGN_YAW"} and isinstance(args, dict):
        return {"tool": tool, "args": {"obj": args.get("obj")}}
    # Unknown shape: return as-is; validator will catch it.
    return obj


@dataclass
class EvalMetrics:
    n: int = 0
    json_valid: int = 0
    schema_valid: int = 0

    tool_exact: int = 0
    tool_confusion: Dict[str, Dict[str, int]] = None  # type: ignore[assignment]

    motion_obj_exact: int = 0
    motion_n: int = 0

    interact_kind_exact: int = 0
    interact_n: int = 0

    interact_choices_len_ok: int = 0

    def __post_init__(self) -> None:
        if self.tool_confusion is None:
            self.tool_confusion = {}


def _bump_confusion(m: EvalMetrics, gt_tool: str, pred_tool: str) -> None:
    row = m.tool_confusion.setdefault(gt_tool, {})
    row[pred_tool] = int(row.get(pred_tool, 0)) + 1


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate model on contract JSONL with tool-level metrics.")
    ap.add_argument("--contract_jsonl", type=str, required=True, help="Path to dataset-contract JSONL (id/instruction/input/output).")
    ap.add_argument("--model_name", type=str, required=True, help="HF model id or local path.")
    ap.add_argument("--adapter_path", type=str, default=None)
    ap.add_argument("--merged_model_path", type=str, default=None)
    ap.add_argument("--use_4bit", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--max_examples", type=int, default=200, help="Max examples to evaluate (sampled).")
    ap.add_argument("--progress_every", type=int, default=25, help="Print progress every N examples.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args(argv)

    set_seed(int(args.seed))
    rng = random.Random(int(args.seed))

    # Load and sample examples (keeps runtime bounded).
    rows = list(_iter_jsonl(str(args.contract_jsonl)))
    if not rows:
        raise SystemExit("Empty contract_jsonl")
    n = min(int(args.max_examples), len(rows))
    sample = rng.sample(rows, n) if n < len(rows) else rows

    # Load model once.
    from .inference import InferenceConfig, _build_messages, _generate_once, _load_model_and_tokenizer

    cfg = InferenceConfig(
        model_name=str(args.model_name),
        adapter_path=str(args.adapter_path) if args.adapter_path else None,
        merged_model_path=str(args.merged_model_path) if args.merged_model_path else None,
        use_4bit=bool(args.use_4bit),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_new_tokens=int(args.max_new_tokens),
        seed=int(args.seed),
        deterministic=True,
    )
    t0_load = time.time()
    model, tok = _load_model_and_tokenizer(cfg)
    print(f"[eval] model loaded in {time.time() - t0_load:.1f}s | evaluating {len(sample)} examples")

    m = EvalMetrics()
    t0 = time.time()

    for idx, r in enumerate(sample, start=1):
        m.n += 1
        instruction = str(r.get("instruction", "")).strip()
        input_str = str(r.get("input", "")).strip()
        gt_str = str(r.get("output", "")).strip()
        # Ground truth tool call object (oracle).
        try:
            gt = json_loads_strict(gt_str)
        except Exception:
            continue
        if not isinstance(gt, dict):
            continue

        prompt = f"{instruction}\n\nInput:\n{input_str}"
        raw = _generate_once(model, tok, _build_messages(prompt), cfg)

        pred_obj, err = _parse_model_json(raw)
        if pred_obj is None:
            _bump_confusion(m, str(gt.get("tool")), "INVALID_JSON")
            continue

        m.json_valid += 1
        pred = _normalize_tool_call(pred_obj)

        # Schema validity (also enforces <=5 choices).
        try:
            validate_tool_call(pred)
            m.schema_valid += 1
        except Exception:
            _bump_confusion(m, str(gt.get("tool")), "INVALID_SCHEMA")

        gt_tool = str(gt.get("tool"))
        pred_tool = str(pred.get("tool"))
        _bump_confusion(m, gt_tool, pred_tool)
        if gt_tool == pred_tool:
            m.tool_exact += 1

        # INTERACT kind + choice length metrics
        if gt_tool == "INTERACT":
            m.interact_n += 1
            gt_kind = str(((gt.get("args") or {}) if isinstance(gt.get("args"), dict) else {}).get("kind"))
            pred_kind = str(((pred.get("args") or {}) if isinstance(pred.get("args"), dict) else {}).get("kind"))
            if gt_kind == pred_kind:
                m.interact_kind_exact += 1
            pred_choices = ((pred.get("args") or {}) if isinstance(pred.get("args"), dict) else {}).get("choices", [])
            if isinstance(pred_choices, list) and len(pred_choices) <= 5:
                m.interact_choices_len_ok += 1

        # Motion obj exactness when both are motion tools.
        if gt_tool in {"APPROACH", "ALIGN_YAW"} and pred_tool in {"APPROACH", "ALIGN_YAW"}:
            m.motion_n += 1
            gt_obj = ((gt.get("args") or {}) if isinstance(gt.get("args"), dict) else {}).get("obj")
            pred_obj_id = ((pred.get("args") or {}) if isinstance(pred.get("args"), dict) else {}).get("obj")
            if gt_obj == pred_obj_id:
                m.motion_obj_exact += 1

        if int(args.progress_every) > 0 and (idx % int(args.progress_every) == 0 or idx == len(sample)):
            dt = max(1e-9, time.time() - t0)
            ex_per_s = idx / dt
            remaining = len(sample) - idx
            eta_s = remaining / max(1e-9, ex_per_s)
            print(f"[eval] {idx}/{len(sample)} examples | {ex_per_s:.2f} ex/s | ETA {eta_s/60:.1f} min")

    # Print a compact summary + JSON for programmatic use.
    def rate(a: int, b: int) -> float:
        return float(a) / float(b) if b else 0.0

    summary = {
        "n": m.n,
        "json_valid_rate": rate(m.json_valid, m.n),
        "schema_valid_rate": rate(m.schema_valid, m.n),
        "tool_exact_rate": rate(m.tool_exact, m.n),
        "motion_obj_exact_rate": rate(m.motion_obj_exact, m.motion_n),
        "interact_kind_exact_rate": rate(m.interact_kind_exact, m.interact_n),
        "interact_choices_len_ok_rate": rate(m.interact_choices_len_ok, m.interact_n),
        "tool_confusion": m.tool_confusion,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


