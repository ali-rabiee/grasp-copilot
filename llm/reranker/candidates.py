"""K-sample candidate generation for the IG reranker.

Generates K candidate INTERACT calls by sampling the LLM at moderate
temperature. The base policy's deterministic call is included as
candidates[0] so InfoGain never makes the policy worse than itself: if
none of the sampled K beats the base call's IG, the selector can pick
the base call back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from llm.utils import json_loads_strict


@dataclass(frozen=True)
class CandidateQuestion:
    tool_call: Dict
    raw_text: str
    sample_idx: int


# Mirrors the noise-sweep runaway fix (commit eeb8ee2): cap past_dialogs in
# every LLM-bound prompt at 12 turns so the prompt never balloons past the
# model's 32k context window during a dialog loop. Used by both the inner
# backend (run_reranker_sweep._build_inner_hf_backend) AND the K-sample
# candidate generation below — both build prompts from input_dict, so both
# need the truncation.
MAX_PAST_DIALOGS = 12


def truncate_past_dialogs(input_dict: Dict[str, Any], max_dialogs: int = MAX_PAST_DIALOGS) -> Dict[str, Any]:
    mem = input_dict.get("memory")
    if not isinstance(mem, dict):
        return input_dict
    dialogs = mem.get("past_dialogs")
    if not isinstance(dialogs, list) or len(dialogs) <= max_dialogs:
        return input_dict
    out = dict(input_dict)
    out["memory"] = dict(mem)
    out["memory"]["past_dialogs"] = list(dialogs)[-max_dialogs:]
    return out


def _parse_first_interact(raw: str) -> Optional[Dict]:
    """Parse raw model output; return the first dict whose top-level tool=="INTERACT"."""
    try:
        obj = json_loads_strict(raw)
    except Exception:
        # Try to extract the first {...} block.
        try:
            from evaluation.benchmarks.offline_exec_benchmark import _extract_first_json_object
            extracted = _extract_first_json_object(raw)
            if not extracted:
                return None
            obj = json_loads_strict(extracted)
        except Exception:
            return None
    if not isinstance(obj, dict):
        return None
    tool = obj.get("tool")
    if isinstance(tool, str) and tool.strip().upper() != "INTERACT":
        return None
    return obj


def _normalize_choices(obj: Dict) -> Dict:
    """Force choices to be a List[str] and clamp to MAX_INTERACT_CHOICES."""
    out = dict(obj)
    args = dict(out.get("args") or {})
    choices = args.get("choices") or []
    if not isinstance(choices, list):
        choices = []
    choices = [str(c) for c in choices][:5]
    args["choices"] = choices
    kind = args.get("kind") or "QUESTION"
    if isinstance(kind, str):
        kind = kind.strip().upper()
    if kind not in {"QUESTION", "SUGGESTION", "CONFIRM"}:
        kind = "QUESTION"
    args["kind"] = kind
    text = args.get("text") or ""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    args["text"] = text
    out["args"] = args
    out["tool"] = "INTERACT"
    return out


def _dedupe(cands: List[CandidateQuestion]) -> List[CandidateQuestion]:
    """Drop candidates whose (kind, sorted choices) collides with an earlier one."""
    out: List[CandidateQuestion] = []
    seen = set()
    for c in cands:
        args = c.tool_call.get("args") or {}
        key = (
            str(args.get("kind", "")).upper(),
            tuple(sorted(str(x) for x in (args.get("choices") or []))),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def generate_candidates(
    input_dict: Dict[str, Any],
    *,
    model,
    tok,
    base_cfg,
    k: int = 5,
    temperature: float = 0.7,
    top_p: float = 0.95,
    seed: int = 0,
    base_call: Optional[Dict] = None,
) -> List[CandidateQuestion]:
    """Sample up to K candidate INTERACT calls.

    Includes `base_call` (the inner backend's deterministic answer) as
    candidate index 0 when it parses as INTERACT, so the reranker can
    always fall back to it if none of the sampled questions has higher IG.

    Skips this entire flow at the call site if `base_call` isn't INTERACT.
    """
    cands: List[CandidateQuestion] = []
    if base_call is not None:
        norm = _normalize_choices(base_call) if base_call.get("tool") == "INTERACT" else base_call
        if norm.get("tool") == "INTERACT":
            cands.append(CandidateQuestion(tool_call=norm, raw_text="<base>", sample_idx=0))

    if k <= len(cands):
        return cands

    # Non-LLM backends (oracle / heuristic) can't sample more candidates —
    # the inner backend is deterministic. Return what we have (typically the
    # base call); the selector picks from a length-1 set.
    if model is None or tok is None or base_cfg is None:
        return cands

    from llm.inference import _build_messages, _generate_once

    instruction = (
        "Given the robot observation and dialog context, infer the user's intent and "
        "emit exactly one tool call. Output ONLY the tool call JSON with keys tool and args. "
        "If the tool is INTERACT, you must output at most 5 choices total."
    )
    truncated_input = truncate_past_dialogs(input_dict)
    prompt = f"{instruction}\n\nInput:\n{json.dumps(truncated_input, ensure_ascii=False)}"
    messages = _build_messages(prompt)

    # Per-sample seed so multiple workers/processes don't collide.
    n_to_draw = k - len(cands)
    samp_cfg = base_cfg.__class__(
        model_path=base_cfg.model_path,
        use_4bit=base_cfg.use_4bit,
        temperature=float(temperature),
        top_p=float(top_p),
        max_new_tokens=base_cfg.max_new_tokens,
        seed=int(seed),
        deterministic=False,
    )

    for i in range(n_to_draw):
        try:
            raw = _generate_once(model, tok, messages, samp_cfg)
        except Exception:
            continue
        parsed = _parse_first_interact(raw)
        if parsed is None:
            continue
        norm = _normalize_choices(parsed)
        if not norm["args"]["choices"]:
            continue
        cands.append(CandidateQuestion(tool_call=norm, raw_text=raw, sample_idx=len(cands)))

    return _dedupe(cands)
