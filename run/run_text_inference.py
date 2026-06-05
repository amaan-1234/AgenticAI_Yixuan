"""Entry point: run the text-LLM ensemble over ChaosNLI (cross-domain track).

    python -m run.run_text_inference --n 20                      # smoke (real GPU)
    python -m run.run_text_inference --n 3113                    # full SNLI+MNLI
    python -m run.run_text_inference --model-index 0 --n 3113    # one model (SLURM array)

Mirrors run_vision_inference: reads config/models_text.yaml, writes
outputs/raw/<key>.jsonl with the SAME record shape, validates dist-sums-to-1 +
structured rationale. Downstream (compute_mta, build_results, report, run_pipeline)
is reused unchanged.
"""
import argparse
import json

import numpy as np

from cac import config
from cac.data import chaosnli
from cac.data.labels_nli import IDX_TO_CLASS, LETTER_TO_IDX, NLI_CLASSES, logprobs_to_dist


def resolve_models(cfg, profile):
    if profile in (None, "active"):
        return cfg["models"]
    profiles = cfg.get("profiles", {})
    if profile not in profiles:
        raise SystemExit(f"unknown profile '{profile}'; have {list(profiles)} + 'active'")
    return profiles[profile]


def _raw_path(key):
    return config.RAW_DIR / f"{key}.jsonl"


def run_one(key, model_id, pairs, runner_kwargs, batch_size):
    from cac.models.text_runner import TextRunner
    runner = TextRunner(model_id, **(runner_kwargs or {}))
    out_path = _raw_path(key)
    n = len(pairs)
    with open(out_path, "w", encoding="utf-8") as f:
        for start in range(0, n, batch_size):
            chunk = pairs[start:start + batch_size]
            letter_lps = runner.distribution_call(chunk)
            rats = runner.rationale_call(chunk)
            for k, (llp, rj) in enumerate(zip(letter_lps, rats)):
                dist = logprobs_to_dist(llp)
                letter = max(llp, key=llp.get) if llp else ""
                rec = {
                    "idx": start + k,
                    "dist": [round(float(x), 6) for x in dist],
                    "letter": letter,
                    "label_dist": IDX_TO_CLASS.get(LETTER_TO_IDX.get(letter, -1), ""),
                    "label_json": rj.get("label", ""),
                    "rationale": rj.get("rationale", {}),
                    "rationale_raw": rj.get("_raw", ""),
                    "parse_status": rj.get("parse_status", ""),
                }
                f.write(json.dumps(rec) + "\n")
            print(f"[infer:{key}] {min(start + batch_size, n)}/{n}", end="\r")
    print()
    runner.close()
    print(f"[infer:{key}] wrote {out_path}")


def validate_one(key, expected_n, tol=0.02):
    from cac.models.schema_nli import RATIONALE_KEYS
    recs = [json.loads(l) for l in open(_raw_path(key), encoding="utf-8")]
    assert len(recs) == expected_n, f"{key}: {len(recs)} != {expected_n}"
    bad = 0
    for r in recs:
        d = r.get("dist")
        if not d or abs(sum(d) - 1.0) > 1e-4:
            raise AssertionError(f"{key} idx {r.get('idx')}: dist sum={sum(d) if d else None}")
        rat = r.get("rationale", {})
        if not (isinstance(rat, dict) and all(k in rat for k in RATIONALE_KEYS)):
            bad += 1
    frac = bad / max(len(recs), 1)
    note = f" ({bad} malformed rationales, {frac:.1%})" if bad else ""
    if frac > tol:
        raise AssertionError(f"{key}: {bad}/{len(recs)} ({frac:.1%}) malformed > tol {tol:.1%}")
    print(f"[validate:{key}] {len(recs)} records OK: dist sums to 1, rationale structured{note}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--split", default="snli_mnli", choices=["snli", "mnli", "snli_mnli"])
    ap.add_argument("--profile", default="active")
    ap.add_argument("--model-index", type=int, default=None)
    ap.add_argument("--skip-if-exists", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    items = chaosnli.load(args.split)[: args.n]
    pairs = [(it.premise, it.hypothesis) for it in items]
    n = len(pairs)

    cfg = config.load_yaml("models_text")
    global_kw = cfg.get("runner", {})
    model_list = resolve_models(cfg, args.profile)
    if args.model_index is not None:
        if not 0 <= args.model_index < len(model_list):
            raise SystemExit(f"--model-index {args.model_index} out of range 0..{len(model_list)-1}")
        model_list = [model_list[args.model_index]]

    for m in model_list:
        if args.skip_if_exists and _raw_path(m["key"]).exists() \
                and sum(1 for _ in open(_raw_path(m["key"]))) >= n:
            print(f"[skip] {m['key']}: already has >= {n} records")
            validate_one(m["key"], n)
            continue
        kw = dict(global_kw)
        for k in ("quantization", "max_model_len", "dtype"):
            if k in m:
                kw[k] = m[k]
        print(f"\n=== {m['key']} ({m['id']}) ===")
        run_one(m["key"], m["id"], pairs, kw, args.batch_size)
        validate_one(m["key"], n)

    print(f"[ok] text inference complete: {[m['key'] for m in model_list]} x {n} NLI pairs")


if __name__ == "__main__":
    main()
