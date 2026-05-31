"""Diagnose why the rationale call produces empty objects.

    python -m run.diagnose_rationales            # real outputs/raw/
    python -m run.diagnose_rationales --mocks    # fixtures under outputs/raw/mocks/

Reads outputs/raw/<model>.jsonl, RE-PARSES each `rationale_raw` with the tolerant
recovery parser (cac.models.json_recovery), and tabulates which step rejects each
empty case. Finishes with a dominant-pattern line and a concrete recommendation —
"deploy the recovery parser" vs "raise max_tokens / flatten schema" — derived from
the actual numbers.
"""

from __future__ import annotations

import argparse
import json
import statistics

from cac import config
from cac.ensemble import inference
from cac.models.json_recovery import _repair_truncated_json, parse_rationale_response
from cac.models.schema import RATIONALE_KEYS


# --- classification -----------------------------------------------------------

def _classify(rec: dict) -> tuple[str, dict, str]:
    """Return (bucket, obj, status_from_parser) for one record.

    Buckets:
      empty_raw           — no raw text at all
      ok                  — strict parse + full rationale
      recovered_full      — needed repair, all 4 rationale keys present
      recovered_partial   — needed repair, rationale missing some keys (truncation)
      no_rationale_field  — parsed but `rationale` is absent or not a dict
      json_error          — parse failed even after repair
    """
    raw = rec.get("rationale_raw", "")
    obj, status = parse_rationale_response(raw)
    if status == "empty":
        return "empty_raw", obj, status
    if status == "json_error":
        return "json_error", obj, status

    rat = obj.get("rationale", {})
    if not isinstance(rat, dict) or not rat:
        return "no_rationale_field", obj, status
    has_all = all(k in rat for k in RATIONALE_KEYS)
    if status == "ok":
        return "ok" if has_all else "no_rationale_field", obj, status
    # recovered:
    return "recovered_full" if has_all else "recovered_partial", obj, status


# --- pretty failure-example printer -------------------------------------------

def _print_failure_examples(bucket: str, records: list[dict], cap: int = 3):
    """Show full raw + json.loads error + ±40 chars context + repair output."""
    print(f"\n  -- {bucket}: first {min(len(records), cap)} of {len(records)} examples --")
    for rec in records[:cap]:
        raw = rec.get("rationale_raw", "")
        print(f"  [idx {rec.get('idx')}]  raw length = {len(raw)}")
        try:
            json.loads(raw)
            err = "<strict json.loads SUCCEEDED — re-check the field-presence path>"
        except json.JSONDecodeError as e:
            pos = e.pos
            ctx_start = max(0, pos - 40)
            ctx_end = min(len(raw), pos + 40)
            ctx = raw[ctx_start:ctx_end].replace("\n", "\\n")
            err = f"{type(e).__name__}: {e.msg} at char {pos} | ...{ctx}..."
        print(f"    json.loads error: {err}")
        # Show the full parse_rationale_response outcome (Strategy 1 + Strategy 2).
        obj, status = parse_rationale_response(raw)
        rat = obj.get("rationale", {}) if isinstance(obj, dict) else {}
        keys_present = ([k for k in RATIONALE_KEYS if k in rat]
                        if isinstance(rat, dict) else [])
        print(f"    parse_rationale_response: status={status}, "
              f"rationale keys present = {keys_present}")
        # Also show what Strategy 1 alone produces (debugging context).
        repaired = _repair_truncated_json(raw)
        if repaired is not None:
            tail_diff = repaired[len(raw):]
            print(f"    (Strategy-1 only appended {tail_diff!r}; "
                  f"Strategy-2 truncate-at-err handles this case)")


# --- per-model output ---------------------------------------------------------

def _print_model(key: str, mock: bool = False):
    base = config.MOCK_RAW_DIR if mock else config.RAW_DIR
    path = base / f"{key}.jsonl"
    if not path.exists():
        print(f"\n[{key}] no JSONL at {path}")
        return None
    recs = inference._read_jsonl(path)
    n = len(recs)
    raw_lens = [len(r.get("rationale_raw", "")) for r in recs]
    raw_mean = statistics.mean(raw_lens) if raw_lens else 0.0

    buckets: dict[str, list[dict]] = {
        "ok": [], "recovered_full": [], "recovered_partial": [],
        "no_rationale_field": [], "json_error": [], "empty_raw": [],
    }
    for r in recs:
        b, _, _ = _classify(r)
        buckets[b].append(r)

    print(f"\n========== {key}  (N={n}, mean raw_len={raw_mean:.0f}) ==========")
    print("  re-parse classification:")
    for b in ("ok", "recovered_full", "recovered_partial",
              "no_rationale_field", "json_error", "empty_raw"):
        cnt = len(buckets[b])
        pct = (cnt / n * 100) if n else 0
        marker = " <-- recovery would fix" if b.startswith("recovered") else ""
        print(f"    {b:<22} {cnt:>4} ({pct:>5.1f}%){marker}")

    # raw_len distribution per bucket — if recovered_partial sits at the
    # max_tokens ceiling while ok/recovered_full sits well below it, the parser
    # is fine and the budget is the bottleneck (raise max_tokens or shorten
    # prompt). If partial and full overlap in length, the model is simply
    # omitting fields and the budget will not help.
    print("  raw_len distribution by bucket:")
    for b in ("ok", "recovered_full", "recovered_partial",
              "no_rationale_field", "json_error"):
        lens = [len(r.get("rationale_raw", "")) for r in buckets[b]]
        if not lens:
            continue
        if len(lens) >= 4:
            q = statistics.quantiles(lens, n=4)
            stats_line = (f"mean={statistics.mean(lens):.0f} "
                          f"min={min(lens)} q25={q[0]:.0f} med={q[1]:.0f} "
                          f"q75={q[2]:.0f} max={max(lens)}")
        else:
            stats_line = (f"mean={statistics.mean(lens):.0f} "
                          f"min={min(lens)} max={max(lens)} (n={len(lens)})")
        print(f"    {b:<22} {stats_line}")

    for b in ("recovered_full", "recovered_partial", "no_rationale_field", "json_error"):
        if buckets[b]:
            _print_failure_examples(b, buckets[b])

    return {b: len(v) for b, v in buckets.items()}


# --- aggregate recommendation -------------------------------------------------

def _print_recommendation(per_model: dict[str, dict[str, int]]):
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)
    tot = {k: 0 for k in ("ok", "recovered_full", "recovered_partial",
                          "no_rationale_field", "json_error", "empty_raw")}
    n = 0
    for stats in per_model.values():
        if not stats:
            continue
        for k, v in stats.items():
            tot[k] += v
        n += sum(stats.values())
    if n == 0:
        print("  no records to summarise.")
        return

    current_empties = tot["recovered_full"] + tot["recovered_partial"] + \
                      tot["no_rationale_field"] + tot["json_error"] + tot["empty_raw"]
    recoverable = tot["recovered_full"] + tot["recovered_partial"]
    print(f"  Total records: {n}")
    print(f"  Currently empty (no usable rationale): {current_empties} "
          f"({current_empties/n*100:.1f}%)")
    print(f"  Recoverable by structural repair:      {recoverable} "
          f"({recoverable/n*100:.1f}%)")
    print(f"    - recovered_full   (all 4 keys):     {tot['recovered_full']}")
    print(f"    - recovered_partial (some keys lost): {tot['recovered_partial']}")
    print(f"  Unrecoverable: json_error={tot['json_error']}, "
          f"no_rationale_field={tot['no_rationale_field']}, empty_raw={tot['empty_raw']}")

    print("\n  RECOMMENDATION:")
    if current_empties == 0 and tot["recovered_partial"] == 0:
        print("    No empty or partial rationales — pipeline is clean. Nothing to do.")
    elif current_empties == 0:
        # All rationales are at least readable, but some are missing fields.
        # Surface this as a separate state so the user knows the parser is
        # fine and the bottleneck is the budget / prompt.
        print("    No EMPTY rationales — parser + recovery are working. But "
              f"{tot['recovered_partial']} record(s) are PARTIAL "
              "(missing tail field like confidence_qualifier).")
        print("    Options to push partial -> ok:")
        print("      (a) raise rationale_call max_tokens 512 -> 640 (smallest "
              "change; works if raw_len is at the ceiling).")
        print("      (b) shorten the rationale prompt for the affected model "
              "family (the verbose-by-default models — InternVL2 in particular).")
        print("    The raw_len distribution above tells you which: if partial "
              "raw_len max ~= budget*4 chars while ok raw_len sits well below, "
              "raise the budget; if they overlap, the model is choosing not to "
              "emit the field and a tighter prompt is the right fix.")
    elif recoverable >= 0.8 * current_empties and recoverable > 0:
        print("    Structural recovery resolves the bulk of failures (>=80%).")
        print("    The runner wiring in this change already deploys it — rerun the")
        print("    WSL2 smoke and the empty rate should drop sharply.")
        if tot["recovered_partial"] > 0.2 * recoverable:
            print("    NOTE: many recoveries are PARTIAL (truncation lost keys). To get")
            print("    full rationales, also raise rationale_call max_tokens above 512.")
    else:
        print("    Structural recovery alone is INSUFFICIENT (<80% of empties).")
        print("    Next moves: (a) raise rationale_call max_tokens (try 640-768) if")
        print("    json_error count is high, (b) consider flattening the schema to")
        print("    {label, key_features, reasoning} if no_rationale_field dominates.")

    # Per-model action items — flag every model whose partial >> full so the
    # caller knows exactly where to apply the fix (e.g. InternVL only).
    partial_dominant = [(k, s) for k, s in per_model.items()
                        if s and s["recovered_partial"] > s["recovered_full"]
                        and s["recovered_partial"] >= 5]
    if partial_dominant:
        print("\n  PER-MODEL action items (partial-dominant):")
        for k, s in partial_dominant:
            total = sum(s.values())
            pct = s["recovered_partial"] / total * 100 if total else 0
            print(f"    [{k}] partial={s['recovered_partial']}/{total} ({pct:.0f}%), "
                  f"full={s['recovered_full']}")
            print(f"      -> first try max_tokens=640 (one-line bump in "
                  f"VLMRunner.rationale_call default).")
            print(f"      -> if still partial after rerun, swap to a shorter "
                  f"prompt for this model family (e.g. drop the "
                  f"'confidence_qualifier' instruction line).")


# --- main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mocks", action="store_true",
                    help="diagnose mock fixtures under outputs/raw/mocks/ instead of real")
    args = ap.parse_args()
    keys = inference.model_keys_from_outputs(mock=args.mocks)
    base = config.MOCK_RAW_DIR if args.mocks else config.RAW_DIR
    if not keys:
        raise SystemExit(f"no outputs in {base}; run inference (or --mock) first")
    per_model = {k: _print_model(k, mock=args.mocks) for k in keys}
    _print_recommendation(per_model)


if __name__ == "__main__":
    main()
