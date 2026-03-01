"""
Offline Executive Benchmark for PRIME paper evaluation.

Evaluates LLM executives on contract JSONL (next tool-call prediction).
Supports: fine-tuned models, zero-shot models, heuristic baselines.

Usage:
    python -m evaluation.offline_exec_benchmark \
        --contract_jsonl data/runs/010/llm_contract.jsonl \
        --models qwen7b_ft=models/qwen2_5_7b_instruct_ft \
        --include_heuristic \
        --include_heuristic_always_ask \
        --out_dir evaluation/eval_outputs/benchmark_run
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from . import _bootstrap  # noqa: F401
except Exception:
    import _bootstrap  # type: ignore  # noqa: F401

from data_generator.grid import manhattan as _grid_manhattan
from data_generator.oracle import validate_tool_call

from llm.inference import InferenceConfig, _build_messages, _generate_once, _load_model_and_tokenizer
from llm.utils import json_loads_strict, set_seed


# =============================================================================
# Utility functions
# =============================================================================

def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _extract_first_json_object(s: str) -> Optional[str]:
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
    """Returns (obj, error)."""
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
    """Normalize model output to oracle schema."""
    tool = obj.get("tool")
    args = obj.get("args")
    tool_norm = tool.strip().upper() if isinstance(tool, str) else tool

    if tool_norm == "INTERACT" and isinstance(args, dict):
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
        choices = [str(c) for c in choices]
        return {"tool": "INTERACT", "args": {"kind": kind, "text": text, "choices": choices}}

    if tool_norm in {"APPROACH", "ALIGN_YAW"} and isinstance(args, dict):
        return {"tool": str(tool_norm), "args": {"obj": args.get("obj")}}

    out = dict(obj)
    if isinstance(tool_norm, str):
        out["tool"] = tool_norm
    return out


def _ctx_bucket(input_json_str: str) -> str:
    """Extract context type from memory.last_prompt.context.type."""
    try:
        inp = json_loads_strict(input_json_str)
    except Exception:
        return "invalid_input_json"
    if not isinstance(inp, dict):
        return "invalid_input_json"
    mem = inp.get("memory")
    if not isinstance(mem, dict):
        return "no_memory"
    last_prompt = mem.get("last_prompt")
    if not isinstance(last_prompt, dict) or not last_prompt:
        return "no_context"
    ctx = last_prompt.get("context")
    if not isinstance(ctx, dict):
        return "no_context"
    t = ctx.get("type")
    if isinstance(t, str) and t.strip():
        return t.strip()
    return "unknown"


def _label_by_id(objects: Sequence[Dict[str, Any]], obj_id: str) -> Optional[str]:
    for o in objects:
        if str(o.get("id")) == str(obj_id):
            lab = o.get("label")
            return str(lab) if isinstance(lab, str) else None
    return None


def _get_num_candidates(input_str: str) -> int:
    """Extract number of candidates from input."""
    try:
        inp = json_loads_strict(input_str)
        mem = inp.get("memory") or {}
        candidates = mem.get("candidates") or []
        return len(candidates)
    except Exception:
        return 0


def _get_user_mode(input_str: str) -> str:
    """Extract user mode from input."""
    try:
        inp = json_loads_strict(input_str)
        user_state = inp.get("user_state") or {}
        return str(user_state.get("mode") or "translation").lower()
    except Exception:
        return "unknown"


# =============================================================================
# Heuristic Baselines
# =============================================================================

def _heuristic_ask_if_ambiguous(input_str: str) -> Dict[str, Any]:
    """
    H1: Ask-if-ambiguous baseline.
    - If 0 candidates -> ask generic
    - If 1 candidate -> act directly
    - If 2+ candidates -> ask candidate menu
    """
    inp = json_loads_strict(input_str)
    assert isinstance(inp, dict)
    objects = inp.get("objects") or []
    memory = inp.get("memory") or {}
    user_state = inp.get("user_state") or {}
    candidates = list((memory.get("candidates") or [])) if isinstance(memory, dict) else []

    mode = str((user_state.get("mode") if isinstance(user_state, dict) else "") or "translation").strip().lower()
    desired_tool = "APPROACH" if mode in {"translation", "gripper"} else "ALIGN_YAW"

    if not candidates:
        return {
            "tool": "INTERACT",
            "args": {"kind": "QUESTION", "text": "I don't see any nearby candidates. What do you want help with?", "choices": ["1) YES", "2) NO"]},
        }

    if len(candidates) == 1:
        return {"tool": desired_tool, "args": {"obj": str(candidates[0])}}

    # Build menu choices
    labels: List[str] = []
    for oid in candidates:
        lab = _label_by_id(objects if isinstance(objects, list) else [], str(oid))
        if lab:
            labels.append(lab)
    labels = labels[:4]
    choices = [f"{i+1}) {lab}" for i, lab in enumerate(labels)]
    if len(choices) < 5:
        choices.append(f"{len(choices)+1}) None of them")
    return {"tool": "INTERACT", "args": {"kind": "QUESTION", "text": "Which object do you want help with?", "choices": choices}}


def _heuristic_always_ask(input_str: str) -> Dict[str, Any]:
    """
    H2: Always-ask baseline.
    Always ask before any motion, even with 1 candidate.
    """
    inp = json_loads_strict(input_str)
    assert isinstance(inp, dict)
    objects = inp.get("objects") or []
    memory = inp.get("memory") or {}
    user_state = inp.get("user_state") or {}
    candidates = list((memory.get("candidates") or [])) if isinstance(memory, dict) else []

    mode = str((user_state.get("mode") if isinstance(user_state, dict) else "") or "translation").strip().lower()
    action_verb = "approach" if mode in {"translation", "gripper"} else "align yaw to"

    if not candidates:
        return {
            "tool": "INTERACT",
            "args": {"kind": "QUESTION", "text": "I don't see any nearby candidates. What do you want help with?", "choices": ["1) YES", "2) NO"]},
        }

    if len(candidates) == 1:
        lab = _label_by_id(objects, str(candidates[0])) or candidates[0]
        return {
            "tool": "INTERACT",
            "args": {"kind": "CONFIRM", "text": f"Do you want me to {action_verb} the {lab}?", "choices": ["1) YES", "2) NO"]},
        }

    # Build menu choices
    labels: List[str] = []
    for oid in candidates:
        lab = _label_by_id(objects if isinstance(objects, list) else [], str(oid))
        if lab:
            labels.append(lab)
    labels = labels[:4]
    choices = [f"{i+1}) {lab}" for i, lab in enumerate(labels)]
    if len(choices) < 5:
        choices.append(f"{len(choices)+1}) None of them")
    return {"tool": "INTERACT", "args": {"kind": "QUESTION", "text": "Which object do you want help with?", "choices": choices}}


def _heuristic_predict_then_assist(input_str: str) -> Dict[str, Any]:
    """
    SA1: Predict-then-Assist baseline.

    Inspired by policy-blending shared autonomy (Dragan & Srinivasa, 2013)
    and predict-then-assist (Herlant et al., 2016).

    Computes per-candidate intent probability via inverse-distance softmax
    weighted by gripper motion direction. Acts when the maximum probability
    exceeds a confidence threshold tau; otherwise asks for clarification.
    """
    _BETA = 1.5          # inverse temperature for distance softmax
    _DIR_W = 0.5         # weight for motion-direction bonus
    _TAU = 0.6           # confidence threshold for acting

    inp = json_loads_strict(input_str)
    assert isinstance(inp, dict)
    objects = inp.get("objects") or []
    memory = inp.get("memory") or {}
    user_state = inp.get("user_state") or {}
    gripper_hist = inp.get("gripper_hist") or []
    candidates = list((memory.get("candidates") or [])) if isinstance(memory, dict) else []

    mode = str((user_state.get("mode") if isinstance(user_state, dict) else "") or "translation").strip().lower()
    desired_tool = "APPROACH" if mode in {"translation", "gripper"} else "ALIGN_YAW"

    if not candidates or not gripper_hist:
        return {
            "tool": "INTERACT",
            "args": {"kind": "QUESTION", "text": "What do you want help with?", "choices": ["1) YES", "2) NO"]},
        }

    if len(candidates) == 1:
        return {"tool": desired_tool, "args": {"obj": str(candidates[0])}}

    current_cell = gripper_hist[-1]["cell"]
    objects_by_id = {str(o["id"]): o for o in objects if isinstance(o, dict)}

    scores: Dict[str, float] = {}
    for cid in candidates:
        obj = objects_by_id.get(str(cid))
        if not obj:
            continue
        dist = _grid_manhattan(current_cell, obj["cell"])
        score = -_BETA * dist

        if len(gripper_hist) >= 2:
            prev_cell = gripper_hist[-2]["cell"]
            prev_dist = _grid_manhattan(prev_cell, obj["cell"])
            score += _DIR_W * (prev_dist - dist)

        scores[str(cid)] = score

    if not scores:
        return {
            "tool": "INTERACT",
            "args": {"kind": "QUESTION", "text": "Which object do you want help with?", "choices": ["1) YES", "2) NO"]},
        }

    max_score = max(scores.values())
    probs = {k: math.exp(v - max_score) for k, v in scores.items()}
    total = sum(probs.values())
    probs = {k: v / total for k, v in probs.items()}

    best_obj = max(probs, key=lambda k: probs[k])
    if probs[best_obj] >= _TAU:
        return {"tool": desired_tool, "args": {"obj": best_obj}}

    labels: List[str] = []
    for oid in candidates:
        lab = _label_by_id(objects if isinstance(objects, list) else [], str(oid))
        if lab:
            labels.append(lab)
    labels = labels[:4]
    choices = [f"{i+1}) {lab}" for i, lab in enumerate(labels)]
    if len(choices) < 5:
        choices.append(f"{len(choices)+1}) None of them")
    return {"tool": "INTERACT", "args": {"kind": "QUESTION", "text": "Which object do you want help with?", "choices": choices}}


def _heuristic_bayesian_intent(input_str: str) -> Dict[str, Any]:
    """
    SA2: Bayesian Intent Inference baseline.

    Inspired by POMDP-based shared autonomy with hindsight optimization
    (Javdani et al., 2015/2018) and POMDP grasping (Yow et al., 2023).

    Maintains a posterior belief over candidate objects using proximity and
    full gripper-motion history as observation evidence.  Uses normalised
    posterior entropy to choose among three strategies:
      - Low  entropy  (<0.3): execute directly on MAP object
      - Mid  entropy  (0.3-0.7): confirm MAP object
      - High entropy  (>0.7): present ranked candidate menu
    """
    _DIST_W = 1.0           # weight for proximity term
    _MOTION_W = 1.5         # weight per history step for motion-toward signal
    _ENT_LOW = 0.3          # below: execute
    _ENT_HIGH = 0.7         # above: full question menu

    inp = json_loads_strict(input_str)
    assert isinstance(inp, dict)
    objects = inp.get("objects") or []
    memory = inp.get("memory") or {}
    user_state = inp.get("user_state") or {}
    gripper_hist = inp.get("gripper_hist") or []
    candidates = list((memory.get("candidates") or [])) if isinstance(memory, dict) else []

    mode = str((user_state.get("mode") if isinstance(user_state, dict) else "") or "translation").strip().lower()
    desired_tool = "APPROACH" if mode in {"translation", "gripper"} else "ALIGN_YAW"
    action_verb = "approach" if mode in {"translation", "gripper"} else "align yaw to"

    if not candidates or not gripper_hist:
        return {
            "tool": "INTERACT",
            "args": {"kind": "QUESTION", "text": "What do you want help with?", "choices": ["1) YES", "2) NO"]},
        }

    if len(candidates) == 1:
        return {"tool": desired_tool, "args": {"obj": str(candidates[0])}}

    current_cell = gripper_hist[-1]["cell"]
    objects_by_id = {str(o["id"]): o for o in objects if isinstance(o, dict)}

    log_liks: Dict[str, float] = {}
    for cid in candidates:
        obj = objects_by_id.get(str(cid))
        if not obj:
            continue
        dist = _grid_manhattan(current_cell, obj["cell"])
        ll = -_DIST_W * dist

        for i in range(1, len(gripper_hist)):
            prev_d = _grid_manhattan(gripper_hist[i - 1]["cell"], obj["cell"])
            cur_d = _grid_manhattan(gripper_hist[i]["cell"], obj["cell"])
            ll += _MOTION_W * (prev_d - cur_d)

        log_liks[str(cid)] = ll

    if not log_liks:
        return {
            "tool": "INTERACT",
            "args": {"kind": "QUESTION", "text": "What do you want help with?", "choices": ["1) YES", "2) NO"]},
        }

    max_ll = max(log_liks.values())
    posteriors = {k: math.exp(v - max_ll) for k, v in log_liks.items()}
    total = sum(posteriors.values())
    posteriors = {k: v / total for k, v in posteriors.items()}

    entropy = -sum(p * math.log(p + 1e-10) for p in posteriors.values())
    max_entropy = math.log(len(posteriors)) if len(posteriors) > 1 else 1.0
    norm_ent = entropy / max_entropy if max_entropy > 0 else 0.0

    map_obj = max(posteriors, key=lambda k: posteriors[k])

    if norm_ent < _ENT_LOW:
        return {"tool": desired_tool, "args": {"obj": map_obj}}

    if norm_ent < _ENT_HIGH:
        lab = _label_by_id(objects, map_obj) or map_obj
        return {
            "tool": "INTERACT",
            "args": {"kind": "CONFIRM", "text": f"Do you want me to {action_verb} the {lab}?", "choices": ["1) YES", "2) NO"]},
        }

    sorted_objs = sorted(posteriors.items(), key=lambda x: -x[1])
    labels: List[str] = []
    for oid, _ in sorted_objs[:4]:
        lab = _label_by_id(objects, str(oid))
        if lab:
            labels.append(lab)
    choices = [f"{i+1}) {lab}" for i, lab in enumerate(labels)]
    if len(choices) < 5:
        choices.append(f"{len(choices)+1}) None of them")
    return {"tool": "INTERACT", "args": {"kind": "QUESTION", "text": "Which object do you want help with?", "choices": choices}}


# =============================================================================
# Metrics
# =============================================================================

@dataclass
class Metrics:
    n: int = 0
    
    # Validity
    json_valid: int = 0
    schema_valid: int = 0
    
    # Level 1: Tool accuracy
    tool_correct: int = 0
    
    # Level 2: Argument correctness
    motion_n: int = 0
    motion_obj_correct: int = 0
    motion_tool_correct: int = 0  # APPROACH vs ALIGN_YAW when both are motion
    
    interact_n: int = 0
    interact_kind_correct: int = 0
    interact_choices_valid: int = 0  # <= 5 choices
    interact_choices_count_correct: int = 0  # Same number of choices
    
    # Strict exact match
    strict_exact: int = 0
    
    # Error breakdowns
    json_errors: int = 0
    schema_errors: int = 0
    
    # Tool type confusion matrix
    tool_confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Per-context breakdown
    by_context: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Per-mode breakdown (translation/rotation/gripper)
    by_mode: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Per-candidate-count breakdown
    by_num_candidates: Dict[str, Dict[str, int]] = field(default_factory=dict)


def _bump(d: Dict[str, int], k: str, inc: int = 1) -> None:
    d[k] = int(d.get(k, 0)) + int(inc)


def _bump_nested(m: Metrics, attr: str, key: str, subkey: str, inc: int = 1) -> None:
    d = getattr(m, attr)
    row = d.setdefault(key, {})
    _bump(row, subkey, inc)


def _rate(a: int, b: int) -> float:
    return float(a) / float(b) if b else 0.0


def _strict_equal(gt: Dict[str, Any], pred: Dict[str, Any], *, ignore_interact_text: bool) -> bool:
    if not ignore_interact_text:
        return gt == pred
    if str(gt.get("tool")) != "INTERACT" or str(pred.get("tool")) != "INTERACT":
        return gt == pred
    gt2 = json.loads(json.dumps(gt))
    pr2 = json.loads(json.dumps(pred))
    if isinstance(gt2.get("args"), dict):
        gt2["args"].pop("text", None)
    if isinstance(pr2.get("args"), dict):
        pr2["args"].pop("text", None)
    return gt2 == pr2


# =============================================================================
# Model Spec
# =============================================================================

@dataclass
class ModelSpec:
    name: str
    model_path: Optional[str] = None
    kind: str = "llm"  # "llm" | "heuristic_ask_if_ambiguous" | "heuristic_always_ask"


# =============================================================================
# Main evaluation function
# =============================================================================

def _eval_one_model(
    spec: ModelSpec,
    rows: List[Dict[str, Any]],
    *,
    seed: int,
    max_examples: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    use_4bit: bool,
    ignore_interact_text_in_strict: bool,
    dump_mistakes_jsonl: Optional[Path],
    max_mistakes: int,
    progress_every: int = 100,
) -> Dict[str, Any]:
    set_seed(int(seed))
    m = Metrics()
    mistakes: List[Dict[str, Any]] = []
    all_predictions: List[Dict[str, Any]] = []

    # Load LLM once per model
    model = tok = None
    cfg = None
    load_s = 0.0
    if spec.kind == "llm":
        if not spec.model_path:
            raise SystemExit(f"Model '{spec.name}' is kind=llm but has no model_path")
        cfg = InferenceConfig(
            model_path=str(spec.model_path),
            use_4bit=bool(use_4bit),
            temperature=float(temperature),
            top_p=float(top_p),
            max_new_tokens=int(max_new_tokens),
            seed=int(seed),
            deterministic=(float(temperature) == 0.0 and float(top_p) == 1.0),
        )
        t0 = time.time()
        model, tok = _load_model_and_tokenizer(cfg)
        load_s = time.time() - t0
        print(f"[{spec.name}] Model loaded in {load_s:.1f}s")

    n = min(int(max_examples), len(rows)) if int(max_examples) > 0 else len(rows)
    sample = rows[:n]
    t0_eval = time.time()

    for idx, r in enumerate(sample, start=1):
        m.n += 1
        ex_id = str(r.get("id", "")).strip()
        instruction = str(r.get("instruction", "")).strip()
        input_str = str(r.get("input", "")).strip()
        gt_str = str(r.get("output", "")).strip()
        
        ctx = _ctx_bucket(input_str)
        user_mode = _get_user_mode(input_str)
        num_cands = _get_num_candidates(input_str)
        cand_bucket = f"cands_{num_cands}" if num_cands <= 5 else "cands_6+"

        _bump_nested(m, "by_context", ctx, "n", 1)
        _bump_nested(m, "by_mode", user_mode, "n", 1)
        _bump_nested(m, "by_num_candidates", cand_bucket, "n", 1)

        # Parse GT
        try:
            gt_obj = json_loads_strict(gt_str)
        except Exception:
            _bump_nested(m, "by_context", ctx, "gt_invalid_json", 1)
            continue
        if not isinstance(gt_obj, dict):
            _bump_nested(m, "by_context", ctx, "gt_not_object", 1)
            continue
        gt = _normalize_tool_call(gt_obj)

        # Predict
        raw = ""
        parse_err = None
        if spec.kind == "heuristic_ask_if_ambiguous":
            pred_raw_obj = _heuristic_ask_if_ambiguous(input_str)
            raw = json.dumps(pred_raw_obj, ensure_ascii=False)
            pred_obj = pred_raw_obj
        elif spec.kind == "heuristic_always_ask":
            pred_raw_obj = _heuristic_always_ask(input_str)
            raw = json.dumps(pred_raw_obj, ensure_ascii=False)
            pred_obj = pred_raw_obj
        elif spec.kind == "heuristic_predict_then_assist":
            pred_raw_obj = _heuristic_predict_then_assist(input_str)
            raw = json.dumps(pred_raw_obj, ensure_ascii=False)
            pred_obj = pred_raw_obj
        elif spec.kind == "heuristic_bayesian_intent":
            pred_raw_obj = _heuristic_bayesian_intent(input_str)
            raw = json.dumps(pred_raw_obj, ensure_ascii=False)
            pred_obj = pred_raw_obj
        else:
            assert cfg is not None and model is not None and tok is not None
            prompt = f"{instruction}\n\nInput:\n{input_str}"
            raw = _generate_once(model, tok, _build_messages(prompt), cfg)
            pred_obj, parse_err = _parse_model_json(raw)
            if pred_obj is None:
                m.json_errors += 1
                _bump_nested(m, "by_context", ctx, "pred_invalid_json", 1)
                _bump_nested(m, "by_mode", user_mode, "pred_invalid_json", 1)
                row = m.tool_confusion.setdefault(str(gt.get("tool")), {})
                _bump(row, "INVALID_JSON", 1)
                if dump_mistakes_jsonl and len(mistakes) < int(max_mistakes):
                    mistakes.append({"id": ex_id, "context": ctx, "mode": user_mode, "num_cands": num_cands, "gt": gt, "pred": None, "raw": raw, "error": parse_err, "error_type": "json_parse"})
                continue

        m.json_valid += 1
        _bump_nested(m, "by_context", ctx, "json_valid", 1)
        _bump_nested(m, "by_mode", user_mode, "json_valid", 1)
        pred = _normalize_tool_call(pred_obj)

        # Schema validity
        try:
            validate_tool_call(pred)
            m.schema_valid += 1
            _bump_nested(m, "by_context", ctx, "schema_valid", 1)
            _bump_nested(m, "by_mode", user_mode, "schema_valid", 1)
        except Exception:
            m.schema_errors += 1
            _bump_nested(m, "by_context", ctx, "schema_invalid", 1)
            _bump_nested(m, "by_mode", user_mode, "schema_invalid", 1)
            row = m.tool_confusion.setdefault(str(gt.get("tool")), {})
            _bump(row, "INVALID_SCHEMA", 1)

        gt_tool = str(gt.get("tool"))
        pred_tool = str(pred.get("tool"))
        row = m.tool_confusion.setdefault(gt_tool, {})
        _bump(row, pred_tool, 1)

        # Level 1: Tool correctness
        tool_ok = gt_tool == pred_tool
        if tool_ok:
            m.tool_correct += 1
            _bump_nested(m, "by_context", ctx, "tool_correct", 1)
            _bump_nested(m, "by_mode", user_mode, "tool_correct", 1)
            _bump_nested(m, "by_num_candidates", cand_bucket, "tool_correct", 1)
        else:
            _bump_nested(m, "by_context", ctx, "tool_wrong", 1)
            _bump_nested(m, "by_mode", user_mode, "tool_wrong", 1)
            _bump_nested(m, "by_num_candidates", cand_bucket, "tool_wrong", 1)

        # Level 2: Conditional argument correctness
        gt_args = gt.get("args") or {}
        pr_args = pred.get("args") or {}
        
        if gt_tool == "INTERACT":
            m.interact_n += 1
            _bump_nested(m, "by_context", ctx, "interact_n", 1)
            
            gt_kind = str(gt_args.get("kind") if isinstance(gt_args, dict) else "")
            pr_kind = str(pr_args.get("kind") if isinstance(pr_args, dict) else "")
            if gt_kind == pr_kind:
                m.interact_kind_correct += 1
                _bump_nested(m, "by_context", ctx, "interact_kind_correct", 1)
            
            pr_choices = pr_args.get("choices", []) if isinstance(pr_args, dict) else []
            if isinstance(pr_choices, list) and len(pr_choices) <= 5:
                m.interact_choices_valid += 1
                _bump_nested(m, "by_context", ctx, "interact_choices_valid", 1)
            
            gt_choices = gt_args.get("choices", []) if isinstance(gt_args, dict) else []
            if len(gt_choices) == len(pr_choices):
                m.interact_choices_count_correct += 1

        if gt_tool in {"APPROACH", "ALIGN_YAW"}:
            m.motion_n += 1
            _bump_nested(m, "by_context", ctx, "motion_n", 1)
            _bump_nested(m, "by_mode", user_mode, "motion_n", 1)
            
            if pred_tool in {"APPROACH", "ALIGN_YAW"}:
                gt_oid = gt_args.get("obj") if isinstance(gt_args, dict) else None
                pr_oid = pr_args.get("obj") if isinstance(pr_args, dict) else None
                if gt_oid == pr_oid:
                    m.motion_obj_correct += 1
                    _bump_nested(m, "by_context", ctx, "motion_obj_correct", 1)
                    _bump_nested(m, "by_mode", user_mode, "motion_obj_correct", 1)
                
                if gt_tool == pred_tool:
                    m.motion_tool_correct += 1
                    _bump_nested(m, "by_context", ctx, "motion_tool_correct", 1)

        # Strict exact match
        if _strict_equal(gt, pred, ignore_interact_text=bool(ignore_interact_text_in_strict)):
            m.strict_exact += 1
            _bump_nested(m, "by_context", ctx, "strict_exact", 1)
            _bump_nested(m, "by_mode", user_mode, "strict_exact", 1)

        # Track mistakes
        if (not tool_ok) and dump_mistakes_jsonl and len(mistakes) < int(max_mistakes):
            mistakes.append({
                "id": ex_id, 
                "context": ctx, 
                "mode": user_mode, 
                "num_cands": num_cands,
                "gt": gt, 
                "pred": pred, 
                "raw": raw[:500] if len(raw) > 500 else raw,
                "error": None,
                "error_type": "tool_mismatch"
            })

        # Progress
        if progress_every > 0 and (idx % progress_every == 0 or idx == len(sample)):
            elapsed = time.time() - t0_eval
            rate = idx / max(elapsed, 1e-6)
            eta = (len(sample) - idx) / max(rate, 1e-6)
            print(f"[{spec.name}] {idx}/{len(sample)} | {rate:.1f} ex/s | ETA {eta/60:.1f}m | tool_acc={_rate(m.tool_correct, m.n):.3f}")

    eval_s = time.time() - t0_eval

    # Build summary
    summary = {
        "model": {"name": spec.name, "kind": spec.kind, "model_path": spec.model_path},
        "timing": {"load_s": load_s, "eval_s": eval_s, "examples_per_sec": m.n / max(eval_s, 1e-6)},
        "n": m.n,
        
        # Validity rates
        "json_valid_rate": _rate(m.json_valid, m.n),
        "schema_valid_rate": _rate(m.schema_valid, m.n),
        "json_errors": m.json_errors,
        "schema_errors": m.schema_errors,
        
        # Primary metrics
        "tool_accuracy": _rate(m.tool_correct, m.n),
        "motion_obj_accuracy": _rate(m.motion_obj_correct, m.motion_n),
        "motion_tool_accuracy": _rate(m.motion_tool_correct, m.motion_n),
        "interact_kind_accuracy": _rate(m.interact_kind_correct, m.interact_n),
        "interact_choices_valid_rate": _rate(m.interact_choices_valid, m.interact_n),
        "strict_exact_rate": _rate(m.strict_exact, m.n),
        
        # Counts
        "motion_n": m.motion_n,
        "interact_n": m.interact_n,
        
        # Breakdowns
        "tool_confusion": m.tool_confusion,
        "by_context": m.by_context,
        "by_mode": m.by_mode,
        "by_num_candidates": m.by_num_candidates,
    }

    if dump_mistakes_jsonl:
        dump_mistakes_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(dump_mistakes_jsonl, "w", encoding="utf-8") as f:
            for row in mistakes:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary["mistakes_path"] = str(dump_mistakes_jsonl)
        summary["mistakes_n"] = len(mistakes)

    return summary


def _parse_model_specs(
    models: Sequence[str],
    *,
    include_heuristic: bool,
    include_heuristic_always_ask: bool,
    include_sa_predict_then_assist: bool = False,
    include_sa_bayesian_intent: bool = False,
) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    for m in models:
        if "=" in m:
            name, path = m.split("=", 1)
            name = name.strip()
            path = path.strip()
            if not name:
                name = Path(path).name
        else:
            path = m.strip()
            name = Path(path).name or path
        specs.append(ModelSpec(name=name, model_path=path, kind="llm"))
    if include_heuristic:
        specs.append(ModelSpec(name="H1_ask_if_ambiguous", kind="heuristic_ask_if_ambiguous"))
    if include_heuristic_always_ask:
        specs.append(ModelSpec(name="H2_always_ask", kind="heuristic_always_ask"))
    if include_sa_predict_then_assist:
        specs.append(ModelSpec(name="SA1_predict_then_assist", kind="heuristic_predict_then_assist"))
    if include_sa_bayesian_intent:
        specs.append(ModelSpec(name="SA2_bayesian_intent", kind="heuristic_bayesian_intent"))
    return specs


# =============================================================================
# Output writers
# =============================================================================

def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = [
        "name", "kind", "model_path", "n",
        "json_valid_rate", "schema_valid_rate",
        "tool_accuracy", "motion_obj_accuracy", "motion_tool_accuracy",
        "interact_kind_accuracy", "interact_choices_valid_rate",
        "strict_exact_rate",
        "motion_n", "interact_n",
        "json_errors", "schema_errors",
        "load_s", "eval_s", "examples_per_sec",
        "mistakes_path", "mistakes_n",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            model = r.get("model") or {}
            timing = r.get("timing") or {}
            out = {
                "name": model.get("name"),
                "kind": model.get("kind"),
                "model_path": model.get("model_path"),
                "n": r.get("n"),
                "json_valid_rate": f"{r.get('json_valid_rate', 0):.4f}",
                "schema_valid_rate": f"{r.get('schema_valid_rate', 0):.4f}",
                "tool_accuracy": f"{r.get('tool_accuracy', 0):.4f}",
                "motion_obj_accuracy": f"{r.get('motion_obj_accuracy', 0):.4f}",
                "motion_tool_accuracy": f"{r.get('motion_tool_accuracy', 0):.4f}",
                "interact_kind_accuracy": f"{r.get('interact_kind_accuracy', 0):.4f}",
                "interact_choices_valid_rate": f"{r.get('interact_choices_valid_rate', 0):.4f}",
                "strict_exact_rate": f"{r.get('strict_exact_rate', 0):.4f}",
                "motion_n": r.get("motion_n"),
                "interact_n": r.get("interact_n"),
                "json_errors": r.get("json_errors"),
                "schema_errors": r.get("schema_errors"),
                "load_s": f"{timing.get('load_s', 0):.1f}",
                "eval_s": f"{timing.get('eval_s', 0):.1f}",
                "examples_per_sec": f"{timing.get('examples_per_sec', 0):.2f}",
                "mistakes_path": r.get("mistakes_path"),
                "mistakes_n": r.get("mistakes_n"),
            }
            w.writerow(out)


def _write_context_breakdown_csv(path: Path, summaries: List[Dict[str, Any]]) -> None:
    """Write per-context accuracy breakdown as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect all contexts across all models
    all_contexts = set()
    for s in summaries:
        all_contexts.update(s.get("by_context", {}).keys())
    contexts = sorted(all_contexts)
    
    fieldnames = ["model"] + [f"{ctx}_acc" for ctx in contexts] + [f"{ctx}_n" for ctx in contexts]
    
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in summaries:
            model_name = s.get("model", {}).get("name", "unknown")
            by_ctx = s.get("by_context", {})
            row = {"model": model_name}
            for ctx in contexts:
                ctx_data = by_ctx.get(ctx, {})
                n = ctx_data.get("n", 0)
                correct = ctx_data.get("tool_correct", 0)
                acc = _rate(correct, n)
                row[f"{ctx}_acc"] = f"{acc:.4f}"
                row[f"{ctx}_n"] = n
            w.writerow(row)


def _write_confusion_matrix_csv(path: Path, summaries: List[Dict[str, Any]]) -> None:
    """Write confusion matrices as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    tools = ["APPROACH", "ALIGN_YAW", "INTERACT", "INVALID_JSON", "INVALID_SCHEMA"]
    
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for s in summaries:
            model_name = s.get("model", {}).get("name", "unknown")
            confusion = s.get("tool_confusion", {})
            w.writerow([f"=== {model_name} ==="])
            w.writerow(["gt\\pred"] + tools)
            for gt_tool in ["APPROACH", "ALIGN_YAW", "INTERACT"]:
                row = [gt_tool]
                for pred_tool in tools:
                    count = confusion.get(gt_tool, {}).get(pred_tool, 0)
                    row.append(count)
                w.writerow(row)
            w.writerow([])


# =============================================================================
# Main
# =============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Offline executive benchmark (multi-model) on contract JSONL.")
    ap.add_argument("--contract_jsonl", type=str, required=True, help="Path to llm_contract*.jsonl.")
    ap.add_argument("--models", type=str, nargs="*", default=[], help='Models: "name=path" or "path".')
    ap.add_argument("--include_heuristic", action="store_true", help="Include H1: ask-if-ambiguous baseline.")
    ap.add_argument("--include_heuristic_always_ask", action="store_true", help="Include H2: always-ask baseline.")
    ap.add_argument("--include_sa_predict_then_assist", action="store_true", help="Include SA1: predict-then-assist baseline.")
    ap.add_argument("--include_sa_bayesian_intent", action="store_true", help="Include SA2: Bayesian intent inference baseline.")
    ap.add_argument("--out_dir", type=str, default="evaluation/eval_outputs/offline_exec", help="Output directory.")

    ap.add_argument("--max_examples", type=int, default=0, help="Max examples (0=all).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress_every", type=int, default=100, help="Print progress every N examples.")

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--use_4bit", action=argparse.BooleanOptionalAction, default=False)

    ap.add_argument("--ignore_interact_text_in_strict", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dump_mistakes", action="store_true", help="Write per-model mistakes JSONL.")
    ap.add_argument("--max_mistakes", type=int, default=200)

    args = ap.parse_args(argv)

    contract_path = str(args.contract_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(_iter_jsonl(contract_path))
    if not rows:
        raise SystemExit("Empty contract_jsonl")
    print(f"[benchmark] Loaded {len(rows)} examples from {contract_path}")

    specs = _parse_model_specs(
        list(args.models),
        include_heuristic=bool(args.include_heuristic),
        include_heuristic_always_ask=bool(args.include_heuristic_always_ask),
        include_sa_predict_then_assist=bool(args.include_sa_predict_then_assist),
        include_sa_bayesian_intent=bool(args.include_sa_bayesian_intent),
    )
    if not specs:
        raise SystemExit("Pass at least one --models path or use --include_heuristic / --include_heuristic_always_ask / --include_sa_predict_then_assist / --include_sa_bayesian_intent.")

    print(f"[benchmark] Evaluating {len(specs)} model(s): {[s.name for s in specs]}")

    all_summaries: List[Dict[str, Any]] = []

    for spec in specs:
        print(f"\n{'='*60}\n[benchmark] Evaluating: {spec.name} (kind={spec.kind})\n{'='*60}")
        mistakes_path = None
        if args.dump_mistakes:
            safe = spec.name.replace("/", "_").replace(" ", "_")
            mistakes_path = out_dir / f"mistakes_{safe}.jsonl"
        summary = _eval_one_model(
            spec,
            rows,
            seed=int(args.seed),
            max_examples=int(args.max_examples),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            max_new_tokens=int(args.max_new_tokens),
            use_4bit=bool(args.use_4bit),
            ignore_interact_text_in_strict=bool(args.ignore_interact_text_in_strict),
            dump_mistakes_jsonl=mistakes_path,
            max_mistakes=int(args.max_mistakes),
            progress_every=int(args.progress_every),
        )
        all_summaries.append(summary)
        
        # Print summary
        print(f"\n[{spec.name}] Results:")
        print(f"  Tool accuracy:        {summary['tool_accuracy']:.4f}")
        print(f"  Motion obj accuracy:  {summary['motion_obj_accuracy']:.4f}")
        print(f"  Interact kind acc:    {summary['interact_kind_accuracy']:.4f}")
        print(f"  Schema valid rate:    {summary['schema_valid_rate']:.4f}")
        print(f"  Strict exact rate:    {summary['strict_exact_rate']:.4f}")

    # Write outputs
    _write_json(out_dir / "summary_all.json", {"contract_jsonl": contract_path, "summaries": all_summaries})
    _write_csv(out_dir / "summary_all.csv", all_summaries)
    _write_context_breakdown_csv(out_dir / "context_breakdown.csv", all_summaries)
    _write_confusion_matrix_csv(out_dir / "confusion_matrices.csv", all_summaries)
    
    print(f"\n[benchmark] Outputs written to: {out_dir}")
    print(f"  - summary_all.json")
    print(f"  - summary_all.csv")
    print(f"  - context_breakdown.csv")
    print(f"  - confusion_matrices.csv")


if __name__ == "__main__":
    main()
