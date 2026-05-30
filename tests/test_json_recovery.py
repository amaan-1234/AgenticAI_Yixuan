"""Tolerant rationale JSON parser (the InternVL truncation fix)."""

import pytest

from cac.models.json_recovery import _repair_truncated_json, parse_rationale_response


# --- strict path ---------------------------------------------------------------

def test_strict_valid_json_returns_ok():
    txt = '{"label": "cat", "rationale": {"key_features": ["fur", "ears"], '\
          '"reasoning": "Cat features visible.", "conclusion": "It is a cat.", '\
          '"confidence_qualifier": "high"}}'
    obj, status = parse_rationale_response(txt)
    assert status == "ok"
    assert obj["label"] == "cat"
    assert obj["rationale"]["confidence_qualifier"] == "high"


# --- recovery path: the actual InternVL failure modes -------------------------

def test_truncated_mid_reasoning_string_recovers():
    """Most common: max_tokens cut off inside the reasoning string."""
    txt = ('{"label": "cat", "rationale": {"key_features": ["fur", "ears"], '
           '"reasoning": "The fur and pointed ears suggest a cat which is a small')
    obj, status = parse_rationale_response(txt)
    assert status == "recovered"
    assert obj["label"] == "cat"
    assert obj["rationale"]["key_features"] == ["fur", "ears"]
    # reasoning ends where the model was cut off
    assert obj["rationale"]["reasoning"].startswith("The fur and pointed ears")
    # truncated keys are simply absent
    assert "conclusion" not in obj["rationale"]
    assert "confidence_qualifier" not in obj["rationale"]


def test_truncated_inside_key_features_list_string_recovers():
    """Cut off mid-list, mid-string."""
    txt = '{"label": "airplane", "rationale": {"key_features": ["wings", "fusela'
    obj, status = parse_rationale_response(txt)
    assert status == "recovered"
    assert obj["label"] == "airplane"
    # the partial string is preserved as the last list item
    assert obj["rationale"]["key_features"][0] == "wings"
    assert obj["rationale"]["key_features"][-1].startswith("fusela")


def test_truncated_after_closed_string_only_missing_braces():
    """String terminates fine but outer braces never close."""
    txt = '{"label": "ship", "rationale": {"key_features": ["hull", "deck"]'
    obj, status = parse_rationale_response(txt)
    assert status == "recovered"
    assert obj["rationale"]["key_features"] == ["hull", "deck"]


def test_internvl_trailing_garbage_after_conclusion_recovers():
    """The observed InternVL2 pattern: complete content + stray quote + whitespace.

    lm-format-enforcer satisfies its schema constraint but emits extra characters
    past the value that strict json.loads rejects ("Expecting ',' delimiter").
    """
    txt = (
        '{"label": "cat", "rationale": {"key_features": ["fur", "ears"], '
        '"reasoning": "Fur and pointed ears.", '
        '"conclusion": "The image is of a cat"\n  \n  " \n  \n  \n\n'
    )
    obj, status = parse_rationale_response(txt)
    assert status == "recovered"
    assert obj["label"] == "cat"
    assert obj["rationale"]["key_features"] == ["fur", "ears"]
    assert obj["rationale"]["conclusion"] == "The image is of a cat"


def test_escaped_quote_inside_string_does_not_confuse_walker():
    txt = '{"label": "dog", "rationale": {"reasoning": "A \\"snout\\" is visible'
    obj, status = parse_rationale_response(txt)
    assert status == "recovered"
    assert "snout" in obj["rationale"]["reasoning"]


# --- failure path --------------------------------------------------------------

def test_non_json_garbage_is_json_error():
    obj, status = parse_rationale_response("hello world, not json at all")
    assert status == "json_error"
    assert obj == {}


def test_empty_input_is_empty_status():
    assert parse_rationale_response("") == ({}, "empty")
    assert parse_rationale_response("   \n  ") == ({}, "empty")


# --- repair function unit checks ----------------------------------------------

def test_repair_returns_none_for_already_balanced():
    assert _repair_truncated_json('{"a": 1}') is None
    assert _repair_truncated_json("[]") is None


def test_repair_appends_quote_then_brackets():
    out = _repair_truncated_json('{"reasoning": "incomplete')
    assert out is not None
    assert out.endswith('"}')
