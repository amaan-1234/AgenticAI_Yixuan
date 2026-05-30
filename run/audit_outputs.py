"""Smoke-test audit (Ask 3): validate the structured-rationale wire-up, no GPU.

    python -m run.audit_outputs

Regenerates mock outputs, then prints a full sample record for one image — soft-label
distribution (10 floats summing to 1), the structured rationale object, and the JSD
between two models — and asserts the on-disk record schema matches what
`cac.ensemble.inference.run_model` writes in --real mode.
"""

import json

import numpy as np

from cac import config
from cac.data import cifar10h
from cac.data.labels import CIFAR10_CLASSES
from cac.ensemble import inference
from cac.ensemble.jsd import _js_divergence, mean_pairwise_jsd
from cac.ensemble.mock import DEFAULT_MODELS, write_mock_outputs
from cac.models.schema import RATIONALE_KEYS

# The contract: keys written by inference.run_model (and mirrored by mock).
EXPECTED_RECORD_KEYS = {"idx", "dist", "letter", "label_dist", "label_json", "rationale",
                        "rationale_raw", "parse_status"}


def main(n: int = 20):
    _, human_probs, _ = cifar10h.prepare()
    write_mock_outputs(human_probs, n=n)
    keys = [k for k, _ in DEFAULT_MODELS]

    # Per-model record lists, aligned by idx (mock outputs live under outputs/raw/mocks/).
    recs_by_model = {k: sorted(inference._read_jsonl(config.MOCK_RAW_DIR / f"{k}.jsonl"),
                               key=lambda r: r["idx"]) for k in keys}

    # Pick the most-disagreeing image among the n for an illustrative example.
    dists, _ = inference.load_distributions(keys, mock=True)          # (M, n, 10)
    ens_jsd = mean_pairwise_jsd(dists)
    img = int(np.argmax(ens_jsd))

    print(f"\n=== AUDIT: image idx {img} (max ensemble JSD of first {n}) ===")
    for k in keys:
        r = recs_by_model[k][img]
        d = np.array(r["dist"], dtype=float)
        soft = {CIFAR10_CLASSES[i]: round(float(d[i]), 4) for i in range(len(d))}
        print(f"\n[{k}]  predicted label: {r['label_dist']}  (json label: {r['label_json']})")
        print(f"  soft_labels (sum={d.sum():.6f}): {soft}")
        print(f"  rationale: {json.dumps(r['rationale'], indent=2)}")
        assert abs(d.sum() - 1.0) < 1e-4, "soft labels do not sum to 1"

    # JSD between the first two models for this image + ensemble value.
    d0 = np.array(recs_by_model[keys[0]][img]["dist"], dtype=float)[None, :]
    d1 = np.array(recs_by_model[keys[1]][img]["dist"], dtype=float)[None, :]
    pair_jsd = float(_js_divergence(d0, d1)[0])
    print(f"\n  JSD({keys[0]} , {keys[1]}) = {pair_jsd:.4f}")
    print(f"  ensemble mean-pairwise JSD       = {ens_jsd[img]:.4f}")

    # Schema check against the run_model contract.
    sample = recs_by_model[keys[0]][img]
    assert set(sample) == EXPECTED_RECORD_KEYS, (
        f"record keys {set(sample)} != run_model contract {EXPECTED_RECORD_KEYS}"
    )
    assert set(sample["rationale"]) == set(RATIONALE_KEYS), (
        f"rationale keys {set(sample['rationale'])} != {set(RATIONALE_KEYS)}"
    )
    print("\n[ok] record schema matches inference.run_model (--real) output:")
    print(f"     record keys = {sorted(EXPECTED_RECORD_KEYS)}")
    print(f"     rationale keys = {list(RATIONALE_KEYS)}")


if __name__ == "__main__":
    main()
