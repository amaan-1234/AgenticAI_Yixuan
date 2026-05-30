"""Guided-decoding schemas for the VLM ensemble.

Two outputs per image per model:
  - distribution call: a single letter A..J (natural logprobs -> soft labels)
  - rationale call: guided JSON {label, rationale{...}} -> the brief's 'valid JSON'
    check plus a STRUCTURED, machine-analysable rationale.

The rationale is a structured object (not free text) so Phase 3 can analyse the
reasoning. `rationale_to_text` flattens it for the cross-encoder MTA, which needs a
single string.
"""

from __future__ import annotations

from cac.data.labels import CIFAR10_CLASSES, LETTERS

RATIONALE_KEYS = ("key_features", "reasoning", "conclusion", "confidence_qualifier")
CONFIDENCE_LEVELS = ["high", "medium", "low"]


def label_json_schema() -> dict:
    """JSON schema for the rationale call. Used as vLLM `guided_json`."""
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": CIFAR10_CLASSES},
            "rationale": {
                "type": "object",
                "properties": {
                    "key_features": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 60},
                        "minItems": 1,
                        "maxItems": 5,
                    },
                    "reasoning": {"type": "string", "maxLength": 300},
                    "conclusion": {"type": "string", "maxLength": 150},
                    "confidence_qualifier": {"type": "string", "enum": CONFIDENCE_LEVELS},
                },
                "required": list(RATIONALE_KEYS),
                "additionalProperties": False,
            },
        },
        "required": ["label", "rationale"],
        "additionalProperties": False,
    }


def letter_choices() -> list[str]:
    """Allowed answers for the distribution call (vLLM `guided_choice` fallback)."""
    return list(LETTERS)


def rationale_to_text(rationale) -> str:
    """Flatten a structured rationale into one string for MTA (cross-encoder).

    Joins key_features + reasoning + conclusion (confidence_qualifier is a category,
    not reasoning, so it is excluded from semantic comparison). Accepts a dict or a
    legacy plain string (returned as-is) for robustness.
    """
    if isinstance(rationale, str):
        return rationale
    if not isinstance(rationale, dict):
        return ""
    feats = rationale.get("key_features", []) or []
    parts = []
    if feats:
        parts.append("Features: " + "; ".join(str(f) for f in feats) + ".")
    if rationale.get("reasoning"):
        parts.append(str(rationale["reasoning"]))
    if rationale.get("conclusion"):
        parts.append(str(rationale["conclusion"]))
    return " ".join(parts).strip()
