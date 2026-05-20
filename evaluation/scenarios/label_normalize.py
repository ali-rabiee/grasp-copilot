"""
Normalize PRIME_LOGS object labels.

The deployed Kinova system uses short labels (`cleanser`, `coffee_can`, `rubik`,
`meat_can`, `mug`, `gelatin_box`) while the YCB training data in
`data_generator/episode.py` uses the full YCB names (`bleach_cleanser`,
`master_chef_can`, `potted_meat_can`, `gelatin_box`, ...).

Scenarios store both:
  * `label`     — the normalized, deployment-side string (what the LLM sees)
  * `raw_label` — the original PRIME_LOGS string, kept verbatim for traceability

We *don't* try to coerce log labels into the YCB vocabulary by default; the
rollout LLM input contract already accepts deployment-side labels, and silently
renaming `rubik` (which is not a YCB object) would be lossy.

Use `to_ycb_label` only when an explicit YCB-side comparison is needed.
"""

from __future__ import annotations

# Deployment-side canonical spellings. Anything not in this map passes through.
_CANONICAL_ALIASES = {
    "cleanser": "cleanser",
    "bleach_cleanser": "cleanser",
    "coffee_can": "coffee_can",
    "master_chef_can": "coffee_can",
    "mug": "mug",
    "meat_can": "meat_can",
    "potted_meat_can": "meat_can",
    "tuna_can": "meat_can",  # observed misclassification in some logs
    "rubik": "rubik",
    "rubiks_cube": "rubik",
    "gelatin_box": "gelatin_box",
    "mustard": "mustard_bottle",
    "mustard_bottle": "mustard_bottle",
}

# Optional reverse map for when a paper table needs YCB-side strings.
_DEPLOY_TO_YCB = {
    "cleanser": "bleach_cleanser",
    "coffee_can": "master_chef_can",
    "mug": "mug",
    "meat_can": "potted_meat_can",
    "gelatin_box": "gelatin_box",
    "mustard_bottle": "mustard_bottle",
    "rubik": "rubik",  # not in YCB; pass through
}


def normalize(raw_label: str) -> str:
    """Map a PRIME_LOGS label to its deployment-side canonical spelling."""
    key = (raw_label or "").strip().lower()
    return _CANONICAL_ALIASES.get(key, key)


def to_ycb_label(canonical_label: str) -> str:
    """Map a deployment-side canonical label to its YCB-side name (if any)."""
    return _DEPLOY_TO_YCB.get(canonical_label, canonical_label)


__all__ = ["normalize", "to_ycb_label"]
