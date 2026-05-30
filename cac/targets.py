"""Ground-truth targets derived from CIFAR-10H human label distributions.

These define what the calibration signal is *measured against*: human entropy
(continuous) and the 'truly hard' mask (top HARD_PCT% by entropy). Centralized so
the pipeline, pre-router, and verification code all use one definition.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import entropy

HARD_PCT = 70  # images at/above this entropy percentile are 'truly hard' (matches v4)
BUDGET = 0.25  # frontier-call budget fraction (matches v4)


def human_entropy(human_probs: np.ndarray) -> np.ndarray:
    """Per-image Shannon entropy of the human label distribution."""
    return entropy(human_probs, axis=1)


def hard_mask(human_probs: np.ndarray, hard_pct: float = HARD_PCT):
    """Return (bool mask of truly-hard images, entropy array, entropy cutoff)."""
    h = human_entropy(human_probs)
    cutoff = np.percentile(h, hard_pct)
    return h >= cutoff, h, float(cutoff)
