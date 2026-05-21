"""Information-gain reranker for PRIME (Package 4).

A post-LLM filter: when the policy decides INTERACT, generate K candidate
questions, score each by expected entropy reduction over the candidate set,
pick argmax. Never overrides ask-vs-act decisions.

Public API:
    make_reranked_backend  — wrap any backend(input_dict)->tool_call
    RerankerConfig         — knobs (k, temperature, prior, selector)
    make_selector          — info_gain / random / oracle / none
    entropy_bits           — exposed for the offline analysis script
    simulate_reply         — exposed for offline replay
"""

from llm.reranker.entropy import entropy_bits, motion_weighted_prior, expected_post_entropy
from llm.reranker.pruning import PruneSnapshot, simulate_reply, infer_pruning_intent
from llm.reranker.selector import (
    Selector,
    InfoGainSelector,
    RandomSelector,
    OracleSelector,
    NoneSelector,
    make_selector,
)
from llm.reranker.policy_wrapper import RerankerConfig, make_reranked_backend

__all__ = [
    "entropy_bits",
    "motion_weighted_prior",
    "expected_post_entropy",
    "PruneSnapshot",
    "simulate_reply",
    "infer_pruning_intent",
    "Selector",
    "InfoGainSelector",
    "RandomSelector",
    "OracleSelector",
    "NoneSelector",
    "make_selector",
    "RerankerConfig",
    "make_reranked_backend",
]
