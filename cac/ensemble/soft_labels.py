"""Convert position-0 token logprobs into a soft-label distribution over 10 classes.

This is the principled replacement for v4's Dirichlet `sim_model`: the distribution
is the model's *actual* probability mass over the class letters, recovered from the
logprobs of a constrained single-token answer. Pure/numpy so it unit-tests without
a GPU or vLLM.
"""

from __future__ import annotations

import numpy as np

from cac.data.labels import LETTER_TO_IDX, LETTERS

K = len(LETTERS)
FLOOR = -20.0  # logprob assigned to class letters absent from the top-k


def logprobs_to_dist(letter_logprobs: dict[str, float], floor: float = FLOOR) -> np.ndarray:
    """{letter: logprob} -> normalized distribution over the 10 classes (np float64, sums to 1)."""
    lp = np.full(K, floor, dtype=np.float64)
    for ltr, v in letter_logprobs.items():
        i = LETTER_TO_IDX.get(str(ltr).strip().upper())
        if i is not None:
            lp[i] = max(lp[i], float(v))
    lp -= lp.max()  # stabilize
    p = np.exp(lp)
    p /= p.sum()
    return p


def stack_distributions(per_image: list[dict[str, float]]) -> np.ndarray:
    """List of {letter: logprob} -> array (N, 10)."""
    return np.vstack([logprobs_to_dist(d) for d in per_image])
