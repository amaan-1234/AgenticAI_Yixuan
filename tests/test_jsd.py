"""JSD equivalence + sanity tests (no GPU)."""

import numpy as np
from scipy.spatial.distance import jensenshannon

from cac.ensemble.jsd import mean_pairwise_jsd


def test_identical_distributions_zero():
    p = np.array([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]])
    dists = np.stack([p, p, p])  # 3 identical models
    assert np.allclose(mean_pairwise_jsd(dists), 0.0, atol=1e-9)


def test_matches_scipy_squared():
    rng = np.random.default_rng(0)
    n, k, m = 50, 10, 3
    dists = np.stack([rng.dirichlet(np.ones(k), size=n) for _ in range(m)])
    ours = mean_pairwise_jsd(dists)
    # reference: mean over pairs of scipy jensenshannon(.)**2
    ref = np.zeros(n)
    cnt = 0
    for i in range(m):
        for j in range(i + 1, m):
            for r in range(n):
                ref[r] += jensenshannon(dists[i][r], dists[j][r]) ** 2
            cnt += 1
    ref /= cnt
    assert np.allclose(ours, ref, atol=1e-8)


def test_disagreement_ordering():
    # two confident-but-opposite models disagree more than two similar ones.
    far = np.stack([np.array([[0.98, 0.01, 0.01]]), np.array([[0.01, 0.01, 0.98]])])
    near = np.stack([np.array([[0.6, 0.3, 0.1]]), np.array([[0.55, 0.35, 0.1]])])
    assert mean_pairwise_jsd(far)[0] > mean_pairwise_jsd(near)[0]
