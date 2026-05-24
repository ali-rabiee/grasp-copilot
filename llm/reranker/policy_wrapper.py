"""Wraps a base backend with the IG reranker.

Contract:
  - inner_backend(input_dict) returns a tool_call dict or None.
  - The wrapper only intercepts inner_backend results whose tool == "INTERACT".
  - Non-INTERACT (motion, None) is returned unchanged — this enforces the
    'reranker does not change ask-vs-act decisions' validation gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence

from llm.reranker.candidates import CandidateQuestion, generate_candidates
from llm.reranker.entropy import information_gain_bits
from llm.reranker.pruning import PruneSnapshot, infer_pruning_intent
from llm.reranker.selector import Selector, make_selector


@dataclass
class RerankerConfig:
    mode: str = "info_gain"           # "none" | "info_gain" | "random" | "oracle"
    k: int = 5
    temperature: float = 0.7
    top_p: float = 0.95
    prior: str = "uniform"            # "uniform" | "motion_weighted"
    seed: int = 0


def make_reranked_backend(
    inner_backend: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
    *,
    model,
    tok,
    base_cfg,
    config: RerankerConfig,
    dialog_log: Optional["DialogLogger"] = None,    # noqa: F821 (forward ref)
    state_hook: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
) -> Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Wrap inner_backend with the IG reranker.

    Args:
        inner_backend: the base policy (LLM/oracle/heuristic).
        model, tok, base_cfg: pre-loaded HF model + tokenizer + InferenceConfig.
            Used to generate K candidates. Ignored if config.mode == "none".
        config: knobs.
        dialog_log: if provided, every reranked INTERACT call appends one
            JSONL record (see evaluation/reranker/dialog_logger.py).
        state_hook: optional callback (input_dict, picked_tool_call) called
            right before returning, useful for OracleSelector to set its
            oracle emit per-call.

    Returns:
        A new backend callable with the same signature as inner_backend.
    """
    selector: Selector = make_selector(config.mode, seed=config.seed)
    is_passthrough = config.mode == "none"

    def backend(input_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raw = inner_backend(input_dict)

        # Passthrough for any non-INTERACT (motion / None / unknown).
        if not isinstance(raw, dict) or raw.get("tool") != "INTERACT":
            return raw

        # mode="none" → return base call untouched, but optionally still log it.
        if is_passthrough:
            if dialog_log is not None:
                snap = PruneSnapshot.from_memory(input_dict.get("memory") or {})
                intent = infer_pruning_intent(raw, snap.candidates, input_dict.get("objects") or [])
                _raw_ch = (raw.get("args") or {}).get("choices")
                n_ch = len(_raw_ch) if isinstance(_raw_ch, (list, tuple)) else 0
                ig, h_b, h_a, per_r = information_gain_bits(
                    intent, n_ch, snap,
                    prior=config.prior,
                    gripper_hist=input_dict.get("gripper_hist") or [],
                    objects=input_dict.get("objects") or [],
                )
                dialog_log.log(
                    input_dict=input_dict,
                    selector_name="none",
                    chosen=raw,
                    chosen_ig=ig,
                    h_before=h_b,
                    h_after_expected=h_a,
                    candidates=[CandidateQuestion(tool_call=raw, raw_text="<base>", sample_idx=0)],
                    per_candidate_scores=[ig],
                    per_candidate_details=[per_r],
                    n_candidates_before=len(snap.candidates),
                    context_type_hint=intent.kind,
                )
            return raw

        # Generate K candidates including the base call at index 0.
        cands = generate_candidates(
            input_dict,
            model=model, tok=tok, base_cfg=base_cfg,
            k=config.k, temperature=config.temperature, top_p=config.top_p,
            seed=config.seed,
            base_call=raw,
        )
        if not cands:
            return raw  # nothing parsed — keep base call

        # Score each candidate's expected information gain.
        snap = PruneSnapshot.from_memory(input_dict.get("memory") or {})
        objects = input_dict.get("objects") or []
        gripper_hist = input_dict.get("gripper_hist") or []

        scores = []
        details = []
        per_reply_all = []
        for c in cands:
            intent = infer_pruning_intent(c.tool_call, snap.candidates, objects)
            _raw_ch = (c.tool_call.get("args") or {}).get("choices")
            n_ch = len(_raw_ch) if isinstance(_raw_ch, (list, tuple)) else 0
            ig, h_b, h_a, per_r = information_gain_bits(
                intent, n_ch, snap,
                prior=config.prior,
                gripper_hist=gripper_hist,
                objects=objects,
            )
            scores.append(ig)
            details.append({"h_before": h_b, "h_after_expected": h_a, "intent": intent.kind})
            per_reply_all.append(per_r)

        idx = selector.pick(cands, scores)
        idx = max(0, min(idx, len(cands) - 1))
        chosen = cands[idx].tool_call

        if dialog_log is not None:
            dialog_log.log(
                input_dict=input_dict,
                selector_name=config.mode,
                chosen=chosen,
                chosen_ig=scores[idx],
                h_before=details[idx]["h_before"],
                h_after_expected=details[idx]["h_after_expected"],
                candidates=cands,
                per_candidate_scores=scores,
                per_candidate_details=per_reply_all,
                n_candidates_before=len(snap.candidates),
                context_type_hint=details[idx]["intent"],
            )

        if state_hook is not None:
            try:
                state_hook(input_dict, chosen)
            except Exception:
                pass
        return chosen

    return backend
