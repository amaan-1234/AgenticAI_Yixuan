"""Reality-check report: real signals vs the v4 simulation benchmarks (Ask 7).

    python -m run.report

Run after build_results (the HPC aggregate job does this automatically). On a
100-image pilot it answers the key question cheaply: do the real VLM soft-label
distributions track human entropy at all? Verdict rule (user's): Pearson r(JSD) > 0.4
=> healthy, scale to 10k; near zero => investigate the prompt / logprob extraction.

Also runnable locally on fixtures (numbers will be simulation-like, not real).
"""

from __future__ import annotations

import json

import numpy as np

from cac import config
from cac.data import cifar10h
from cac.ensemble import inference
from cac.ensemble.jsd import mean_pairwise_jsd
from cac.models.schema import rationale_to_text
from cac.pipeline import weights
from cac.pipeline.metrics import correlations, normalise_01
from cac.targets import human_entropy as entropy_of

# v4 simulation benchmarks (from the lit review / test_disagreement_v4.py).
V4 = {"r_jsd": 0.668, "rho_jsd": 0.816, "r_dual": 0.796, "frontier_pct": 0.25}


def _validation_failures(keys):
    """Per-model counts of invalid distributions / empty rationales (non-raising)."""
    out = {}
    for key in keys:
        recs = inference._read_jsonl(config.RAW_DIR / f"{key}.jsonl")
        bad_dist = sum(1 for r in recs
                       if not r.get("dist") or abs(sum(r["dist"]) - 1.0) > 1e-4)
        empty_rat = sum(1 for r in recs
                        if not rationale_to_text(r.get("rationale", {})).strip())
        out[key] = {"n": len(recs), "bad_dist": bad_dist, "empty_rationale": empty_rat}
    return out


def _load_mta(n):
    if config.MTA_SCORES.exists():
        m = np.load(config.MTA_SCORES)
        if len(m) == n:
            return m
    return None


def _escalation_rate():
    path = config.OUTPUTS_DIR / "results.jsonl"
    if not path.exists():
        return None
    seen = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            seen[r["image_id"]] = bool(r["escalate"])
    return float(np.mean(list(seen.values()))) if seen else None


def _row(label, real, bench, fmt="{:.4f}"):
    rs = fmt.format(real) if real is not None else "  n/a"
    bs = fmt.format(bench) if bench is not None else "  n/a"
    delta = f"{real - bench:+.4f}" if (real is not None and bench is not None) else "   -"
    print(f"  {label:<28} {rs:>10} {bs:>12} {delta:>10}")


def main():
    dists, keys = inference.load_distributions()
    m, n, _ = dists.shape
    _, human_probs, _ = cifar10h.prepare()
    he = entropy_of(human_probs[:n])

    ensemble_jsd = mean_pairwise_jsd(dists)
    r_jsd, rho_jsd = correlations(ensemble_jsd, he)

    mta = _load_mta(n)
    r_dual = None
    if mta is not None:
        jsd_norm = normalise_01(ensemble_jsd)
        one_minus_mta = 1.0 - mta
        w = weights.mi_weights(jsd_norm, one_minus_mta, he)
        escalation = w["alpha"] * jsd_norm + w["beta"] * one_minus_mta
        r_dual, _ = correlations(escalation, he)

    esc_rate = _escalation_rate()
    vfail = _validation_failures(keys)

    bar = "=" * 64
    print(f"\n{bar}\nREALITY-CHECK REPORT  (N={n}, models={keys})\n{bar}")
    print(f"  {'metric':<28} {'REAL':>10} {'v4 sim':>12} {'delta':>10}")
    print("  " + "-" * 60)
    _row("Pearson r (JSD vs H_human)", r_jsd, V4["r_jsd"])
    _row("Spearman rho (JSD)", rho_jsd, V4["rho_jsd"])
    _row("Pearson r (dual signal)", r_dual, V4["r_dual"])
    if esc_rate is not None:
        _row("Escalation rate", esc_rate, V4["frontier_pct"], fmt="{:.3f}")

    if mta is not None:
        q = np.percentile(mta, [0, 25, 50, 75, 100])
        print("\n  MTA distribution:")
        print(f"    mean={mta.mean():.3f} std={mta.std():.3f} "
              f"min={q[0]:.3f} q25={q[1]:.3f} med={q[2]:.3f} q75={q[3]:.3f} max={q[4]:.3f}")
    else:
        print("\n  MTA distribution: (no MTA cache matching N — run compute_mta)")

    print("\n  Validation failures (per model):")
    for key, v in vfail.items():
        print(f"    {key:<22} records={v['n']:>6}  bad_dist={v['bad_dist']:>4}  "
              f"empty_rationale={v['empty_rationale']:>4}")

    print("\n  " + "-" * 60)
    if r_jsd > 0.4:
        print(f"  VERDICT: HEALTHY (r_jsd={r_jsd:.3f} > 0.4) -> scale to the full 10k run.")
    else:
        print(f"  VERDICT: INVESTIGATE (r_jsd={r_jsd:.3f} <= 0.4) -> check the prompt / "
              f"logprob extraction / image preprocessing before scaling.")
    print(bar)


if __name__ == "__main__":
    main()
