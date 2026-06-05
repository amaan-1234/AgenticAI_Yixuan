"""Guided-decoding schema for the 3-class NLI rationale call (analogue of schema.py).
Reuses the same structured-rationale shape so MTA / build_results / report consume
text outputs unchanged. rationale_to_text is imported from the vision schema.
"""
from __future__ import annotations
from cac.data.labels_nli import NLI_CLASSES, LETTERS

RATIONALE_KEYS = ("key_features", "reasoning", "conclusion", "confidence_qualifier")
CONFIDENCE_LEVELS = ["high", "medium", "low"]

def label_json_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": NLI_CLASSES},
            "rationale": {
                "type": "object",
                "properties": {
                    "key_features": {"type": "array",
                        "items": {"type": "string", "maxLength": 80},
                        "minItems": 1, "maxItems": 5},
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
    return list(LETTERS)
