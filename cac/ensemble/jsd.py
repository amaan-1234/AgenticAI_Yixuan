"""Vectorized mean-pairwise Jensen-Shannon divergence across the ensemble.

Replaces v4's triple-nested `mean_pairwise_jsd` (which called
scipy.spatial.distance.jensenshannon per image per pair and squared it). scipy's
jensenshannon returns the JS *distance* (sqrt of the divergence, natural log), so
its square equals the JS divergence computed here — identical semantics, ~100x faster.

Input:  dists (M, N, K) — M model distributions over N items, K classes.
Output: (N,) mean pairwise JS divergence per item (the primary disagreement signal).
"""

from __future__ import annotations

import numpy as np


def _js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Row-wise JS divergence (natural log) for p,q of shape (N, K)."""
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)), axis=1)
    kl_qm = np.sum(q * (np.log(q) - np.log(m)), axis=1)
    return 0.5 * kl_pm + 0.5 * kl_qm


def mean_pairwise_jsd(dists: np.ndarray) -> np.ndarray:
    """dists (M, N, K) -> (N,) mean pairwise JS divergence."""
    dists = np.asarray(dists, dtype=np.float64)
    m = dists.shape[0]
    assert m >= 2, "need >=2 models to measure disagreement"
    agg = np.zeros(dists.shape[1], dtype=np.float64)
    cnt = 0
    for i in range(m):
        for j in range(i + 1, m):
            agg += _js_divergence(dists[i], dists[j])
            cnt += 1
    return agg / cnt
