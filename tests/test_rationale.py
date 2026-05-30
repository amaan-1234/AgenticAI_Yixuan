"""Structured rationale schema + serialization contract (no GPU)."""

import json

import pytest

from cac import config
from cac.ensemble import inference
from cac.models.schema import (
    CONFIDENCE_LEVELS,
    RATIONALE_KEYS,
    label_json_schema,
    rationale_to_text,
)


def _good_record(idx: int = 0) -> dict:
    return {
        "idx": idx,
        "dist": [0.1] * 10,
        "letter": "A",
        "label_dist": "airplane",
        "label_json": "airplane",
        "rationale": {
            "key_features": ["wings"],
            "reasoning": "Wings imply a plane.",
            "conclusion": "It is an airplane.",
            "confidence_qualifier": "high",
        },
    }


def _write_records(key: str, records: list[dict]):
    path = config.RAW_DIR / f"{key}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def test_schema_rationale_is_structured_object():
    schema = label_json_schema()
    rat = schema["properties"]["rationale"]
    assert rat["type"] == "object"
    assert set(rat["required"]) == set(RATIONALE_KEYS)
    assert rat["properties"]["key_features"]["type"] == "array"
    assert rat["properties"]["confidence_qualifier"]["enum"] == CONFIDENCE_LEVELS


def test_rationale_to_text_joins_fields():
    r = {
        "key_features": ["beak", "feathers"],
        "reasoning": "Beak and feathers imply a bird.",
        "conclusion": "It is a bird.",
        "confidence_qualifier": "high",
    }
    txt = rationale_to_text(r)
    assert "beak" in txt and "feathers" in txt
    assert "imply a bird" in txt and "It is a bird." in txt
    assert "high" not in txt  # confidence excluded from semantic text


def test_rationale_to_text_handles_legacy_and_garbage():
    assert rationale_to_text("plain string") == "plain string"
    assert rationale_to_text(None) == ""
    assert rationale_to_text({}) == ""


# --- validate_model_output: the gate the SLURM array task uses ----------------

@pytest.mark.parametrize("missing", ["key_features", "reasoning", "conclusion",
                                     "confidence_qualifier"])
def test_validate_rejects_missing_rationale_key(missing):
    key = f"_test_missing_{missing}"
    rec = _good_record()
    rec["rationale"].pop(missing)
    path = _write_records(key, [rec])
    try:
        with pytest.raises(AssertionError, match="malformed rationales"):
            inference.validate_model_output(key, 1)
    finally:
        path.unlink(missing_ok=True)


def test_validate_rejects_bad_distribution():
    key = "_test_bad_dist"
    rec = _good_record()
    rec["dist"] = [0.5] * 10  # sums to 5, not 1
    path = _write_records(key, [rec])
    try:
        with pytest.raises(AssertionError, match="distribution invalid"):
            inference.validate_model_output(key, 1)
    finally:
        path.unlink(missing_ok=True)


def test_validate_accepts_good_record():
    key = "_test_good"
    path = _write_records(key, [_good_record()])
    try:
        assert inference.validate_model_output(key, 1) is True
    finally:
        path.unlink(missing_ok=True)
