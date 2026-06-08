"""Frontier-model baseline + uncertainty control on the ChaosNLI subset.

Reads the frozen stratified subset (outputs/chaosnli_frontier_subset.json) and asks
a Claude model for its CLASS PROBABILITY DISTRIBUTION over the three NLI labels in a
single structured call per item. Reports:
  1. frontier accuracy vs human-majority  (the accuracy ceiling)
  2. corr(frontier uncertainty, H_human)  (the CONTROL: does a strong model's
     stated uncertainty track human disagreement where the small ensemble's did not?)

Why distributions, not repeated letter-sampling: at temperature the model returns the
same forced letter on nearly every draw (verified empirically -> zero variance), so
sampling yields no uncertainty signal. Asking the model for an explicit distribution
elicits its uncertainty directly and costs one call per item instead of five. This
mirrors how the ChaosNLI literature compares model distributions to human ones.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...        # NEVER hardcode the key
    python -m run.run_frontier --limit 5       # smoke test
    python -m run.run_frontier                 # full 450

Caching: per-item results append to outputs/frontier_chaosnli.jsonl and are flushed
immediately; on restart, items already present are skipped -> no re-billing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import numpy as np

from cac import config
from cac.data.labels_nli import NLI_CLASSES

SUBSET_PATH = config.OUTPUTS_DIR / "chaosnli_frontier_subset.json"
OUT_PATH = config.OUTPUTS_DIR / "frontier_chaosnli.jsonl"
DEFAULT_MODEL = "claude-sonnet-4-6"

PROMPT = (
    "You are an expert annotator for natural language inference. Given a premise and "
    "hypothesis, humans often disagree, so express your judgment as a probability "
    "distribution over the three labels rather than a single answer.\n\n"
    "Premise: {premise}\nHypothesis: {hypothesis}\n\n"
    "Output ONLY a JSON object with three keys that sum to 1.0, e.g. "
    '{{"entailment": 0.6, "neutral": 0.3, "contradiction": 0.1}}. '
    "Reflect genuine uncertainty: if the relationship is ambiguous, spread the mass; "
    "if it is clear, concentrate it. Output only the JSON, nothing else."
)


def _client():
    try:
        import anthropic
    except ImportError:
        raise SystemExit("pip install anthropic --break-system-packages")
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY not set. Run: export ANTHROPIC_API_KEY=sk-ant-...")
    return anthropic.Anthropic(api_key=key)


def _parse_dist(text: str) -> np.ndarray | None:
    """Pull a {entailment,neutral,contradiction} JSON object from the reply -> (3,) prob vector."""
    m = re.search(r"\{[^}]*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    try:
        v = np.array([float(obj[c]) for c in NLI_CLASSES], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    s = v.sum()
    if not np.isfinite(s) or s <= 0:
        return None
    return v / s


def _call(client, model, premise, hypothesis, temperature, max_retries=5):
    msg = PROMPT.format(premise=premise, hypothesis=hypothesis)
    for attempt in range(max_retries):
        try:
            kwargs = dict(model=model, max_tokens=60,
                          messages=[{"role": "user", "content": msg}])
            if temperature is not None:
                kwargs["temperature"] = temperature
            resp = client.messages.create(**kwargs)
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            return _parse_dist(text), text
        except Exception as e:
            wait = 2 ** attempt
            print(f"    [retry {attempt+1}/{max_retries} in {wait}s] {type(e).__name__}: {str(e)[:120]}")
            time.sleep(wait)
    return None, ""


def _entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _load_done() -> set:
    if not OUT_PATH.exists():
        return set()
    done = set()
    with open(OUT_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["uid"])
            except Exception:
                continue
    return done


def run(model, limit, temperature):
    items = json.load(open(SUBSET_PATH, encoding="utf-8"))
    if limit:
        items = items[:limit]
    done = _load_done()
    client = _client()
    print(f"[frontier] model={model} mode=probs temp={temperature} "
          f"items={len(items)} (already done: {len(done)})")

    with open(OUT_PATH, "a", encoding="utf-8") as f:
        for n_done, it in enumerate(items, 1):
            if it["uid"] in done:
                continue
            dist, raw = _call(client, model, it["premise"], it["hypothesis"], temperature)
            if dist is None:
                dist = np.full(len(NLI_CLASSES), 1.0 / len(NLI_CLASSES))
                parse_ok = False
            else:
                parse_ok = True
            rec = {
                "uid": it["uid"], "idx": it["idx"],
                "frontier_dist": [round(float(x), 4) for x in dist],
                "frontier_argmax": int(dist.argmax()),
                "frontier_entropy_bits": round(_entropy_bits(dist), 4),
                "parse_ok": parse_ok,
                "raw": raw[:200],
                "human_dist": it["human_dist"],
                "human_entropy": it["entropy"],
                "human_argmax": int(np.argmax(it["human_dist"])),
            }
            f.write(json.dumps(rec) + "\n"); f.flush()
            if n_done % 20 == 0:
                print(f"  [{n_done}/{len(items)}] uid={it['uid']} dist={rec['frontier_dist']} ok={parse_ok}")
    analyze()


def analyze():
    recs = [json.loads(l) for l in open(OUT_PATH, encoding="utf-8")]
    if not recs:
        print("[frontier] no results yet."); return
    bad = sum(1 for r in recs if not r.get("parse_ok", True))
    fa = np.array([r["frontier_argmax"] for r in recs])
    ha = np.array([r["human_argmax"] for r in recs])
    H = np.array([r["human_entropy"] for r in recs])
    fH = np.array([r["frontier_entropy_bits"] for r in recs])

    print("\n" + "=" * 60)
    print(f"FRONTIER BASELINE  (N={len(recs)}, parse failures={bad})")
    print("=" * 60)
    print(f"  accuracy vs human-majority           : {(fa==ha).mean():.3f}")
    print(f"    (small-model subset: qwen 0.529, gemma 0.571)")
    if fH.std() > 1e-9:
        r = float(np.corrcoef(fH, H)[0, 1])
        print(f"  corr(frontier entropy, H_human)      : {r:.3f}")
        print(f"    (small-ensemble disagreement-vs-H was 0.046 -> if this is")
        print(f"     clearly higher, the NLI failure is small-model-specific)")
    else:
        print(f"  corr(frontier entropy, H): n/a (frontier gave zero-variance distributions)")
    order = np.argsort(H)
    t = len(order) // 3
    for name, sl in [("low-H", order[:t]), ("mid-H", order[t:2*t]), ("high-H", order[2*t:])]:
        if len(sl) == 0:
            continue
        print(f"    acc[{name}] = {(fa[sl]==ha[sl]).mean():.3f}  (mean H={H[sl].mean():.3f}, "
              f"mean frontier entropy={fH[sl].mean():.3f})")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0.0 is fine; we want the model's stated distribution, not sampled answers")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    temp = None if "opus" in args.model.lower() else args.temperature
    if args.analyze_only:
        analyze()
    else:
        run(args.model, args.limit, temp)


if __name__ == "__main__":
    main()