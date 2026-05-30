"""Consolidated, crash-safe results database (Ask 4).

Joins per-model inference outputs with the computed signals and the final escalation
decision into one JSON-lines file (`outputs/results.jsonl`), one row per (image, model):

    {image_id, model_name, soft_labels:{class:prob}, rationale:{...},
     jsd_vs_others:float, human_entropy:float, escalate:bool}

Rows are written incrementally (append + flush) so a job that dies mid-run keeps every
completed record. Regenerable from outputs/raw/*.jsonl at any time.
"""

from __future__ import annotations

import json
import os

import numpy as np
from sklearn.model_selection import train_test_split

from cac import config
from cac.data import cifar10h
from cac.data.labels import CIFAR10_CLASSES
from cac.ensemble import inference
from cac.ensemble.jsd import _js_divergence, mean_pairwise_jsd
from cac.pipeline import calibration, weights
from cac.pipeline.metrics import normalise_01
from cac.targets import BUDGET, hard_mask, human_entropy as entropy_of


class ResultsWriter:
    """Incremental JSON-lines writer (append + flush, optional fsync)."""

    def __init__(self, path=config.OUTPUTS_DIR / "results.jsonl", fsync: bool = False):
        self.path = path
        self.fsync = fsync
        self._f = open(path, "w", encoding="utf-8")
        self.n = 0

    def append(self, record: dict):
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()
        if self.fsync:
            os.fsync(self._f.fileno())
        self.n += 1

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _jsd_vs_others(dists: np.ndarray) -> np.ndarray:
    """(M,N,K) -> (M,N): for each model, mean JS divergence vs every other model."""
    m, n, _ = dists.shape
    out = np.zeros((m, n))
    for i in range(m):
        acc = np.zeros(n)
        for j in range(m):
            if i != j:
                acc += _js_divergence(dists[i], dists[j])
        out[i] = acc / max(m - 1, 1)
    return out


def _load_or_compute_mta(n: int, model_keys) -> np.ndarray:
    """Use cached MTA if it matches N, else compute from rationale text."""
    if config.MTA_SCORES.exists():
        cached = np.load(config.MTA_SCORES)
        if len(cached) == n:
            return cached
    from cac.mta.cross_encoder import MTAScorer

    return MTAScorer().mta_for_rationales(inference.load_rationales(model_keys))


def build_results(model_keys=None, out_path=config.OUTPUTS_DIR / "results.jsonl") -> str:
    dists, keys = inference.load_distributions(model_keys)        # (M,N,K)
    rationale_objs = inference.load_rationale_objects(keys)        # [N][M]
    m, n, _ = dists.shape

    _, human_probs, _ = cifar10h.prepare()
    human_probs = human_probs[:n]
    he = entropy_of(human_probs)
    truly_hard, _, _ = hard_mask(human_probs)

    jvo = _jsd_vs_others(dists)                                    # (M,N)
    ensemble_jsd = mean_pairwise_jsd(dists)
    mta = _load_or_compute_mta(n, keys)

    # Escalation decision: dual signal -> calibrated P(hard) -> threshold.
    jsd_norm = normalise_01(ensemble_jsd)
    one_minus_mta = 1.0 - mta
    w = weights.mi_weights(jsd_norm, one_minus_mta, he)
    escalation = w["alpha"] * jsd_norm + w["beta"] * one_minus_mta
    idx_tr, idx_va = train_test_split(np.arange(n), test_size=0.3, random_state=42)
    cal = calibration.fit_calibrators(escalation, truly_hard, idx_tr, idx_va, BUDGET)
    p_all = cal["best"]["predict"](escalation)
    escalate = p_all >= cal["best"]["tau"]

    with ResultsWriter(out_path) as w_out:
        for i in range(n):
            for mi, key in enumerate(keys):
                d = dists[mi, i]
                w_out.append({
                    "image_id": int(i),
                    "model_name": key,
                    "soft_labels": {CIFAR10_CLASSES[c]: round(float(d[c]), 6)
                                    for c in range(len(d))},
                    "rationale": rationale_objs[i][mi],
                    "jsd_vs_others": round(float(jvo[mi, i]), 6),
                    "human_entropy": round(float(he[i]), 6),
                    "escalate": bool(escalate[i]),
                })

    print(f"[results] wrote {w_out.n} rows ({n} images x {m} models) -> {out_path}")
    print(f"[results] calibrator={cal['best']['name']} tau={cal['best']['tau']:.3f} "
          f"escalated images={int(escalate.sum())}/{n} ({escalate.mean()*100:.1f}%)")
    return str(out_path)
