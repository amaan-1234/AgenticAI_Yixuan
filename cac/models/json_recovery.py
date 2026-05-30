"""Tolerant JSON parsing for VLM rationale responses.

Designed for the InternVL2 failure mode: well-formed JSON-shaped text whose tail
got truncated at `max_tokens`, leaving an unclosed string and/or unclosed braces.
A single-pass structural walk closes them, then `json.loads` succeeds with the
content the model actually produced (anything past the truncation point simply
isn't there). The recovery is purely structural — it doesn't fabricate field
values, so `validate_model_output`'s rationale-key check still catches genuinely
broken outputs.
"""

from __future__ import annotations

import json


def _repair_truncated_json(txt: str) -> str | None:
    """Close any unterminated string + unclosed `{`/`[` at the end. None if nothing to do."""
    in_string = False
    escape_next = False
    stack: list[str] = []
    for ch in txt:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{" or ch == "[":
            stack.append(ch)
        elif ch == "}":
            if stack and stack[-1] == "{":
                stack.pop()
        elif ch == "]":
            if stack and stack[-1] == "[":
                stack.pop()

    if not in_string and not stack:
        return None  # nothing to repair; strict parse should already have worked

    repaired = txt
    if in_string:
        repaired += '"'
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"
    return repaired


def parse_rationale_response(txt: str) -> tuple[dict, str]:
    """Parse a model's rationale JSON tolerantly.

    Two recovery strategies, applied in order:
      1. Close-on-full: walk the whole input, close any unterminated string +
         unclosed braces, retry. Handles pure truncation (end-of-input mid-string).
      2. Truncate-at-error + close: when strict json.loads fails at position P,
         take the prefix `txt[:P]`, close unclosed structures, retry. Handles the
         observed InternVL2 pattern: lm-format-enforcer emits a complete JSON value
         then a stray `"` + whitespace, breaking the parser past the valid prefix.

    Returns (obj, status) where status is one of:
      'empty'      — input was blank
      'ok'         — strict json.loads succeeded
      'recovered'  — strict failed but one of the recoveries succeeded
      'json_error' — all attempts failed; obj is {}
    """
    if not txt or not txt.strip():
        return {}, "empty"
    try:
        return json.loads(txt), "ok"
    except json.JSONDecodeError as e:
        err_pos = e.pos

    # Strategy 1: full-input close (pure truncation case).
    repaired = _repair_truncated_json(txt)
    if repaired is not None:
        try:
            return json.loads(repaired), "recovered"
        except json.JSONDecodeError:
            pass

    # Strategy 2: truncate at strict-parse error, then close (corruption-after-prefix).
    if 0 < err_pos < len(txt):
        prefix = txt[:err_pos]
        rep_prefix = _repair_truncated_json(prefix)
        candidate = rep_prefix if rep_prefix is not None else prefix
        try:
            return json.loads(candidate), "recovered"
        except json.JSONDecodeError:
            pass

    return {}, "json_error"
