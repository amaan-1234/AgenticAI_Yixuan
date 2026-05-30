"""MI-based alpha/beta tuning for the dual-signal escalation formula (v4 section 5).

    Escalation = alpha * JSD_norm + beta * (1 - MTA)
    alpha = MI(JSD, H_human) / (MI(JSD, H_human) + MI(1-MTA, H_human))

MI-derived weights are primary; logistic-regression coefficients are the fallback
for small/noisy calibration sets. Logic unchanged from v4.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score

N_MI_BINS = 30


def continuous_mi(x: np.ndarray, y: np.ndarray, n_bins: int = N_MI_BINS) -> float:
    """MI between two continuous variables via equal-width binning."""
    x_bins = np.digitize(x, np.linspace(x.min(), x.max(), n_bins + 1)[:-1])
    y_bins = np.digitize(y, np.linspace(y.min(), y.max(), n_bins + 1)[:-1])
    return mutual_info_score(x_bins, y_bins)


def mi_weights(jsd_norm: np.ndarray, one_minus_mta: np.ndarray,
               human_entropy: np.ndarray) -> dict:
    mi_jsd = continuous_mi(jsd_norm, human_entropy)
    mi_mta = continuous_mi(one_minus_mta, human_entropy)
    total = mi_jsd + mi_mta + 1e-12
    return {"alpha": mi_jsd / total, "beta": mi_mta / total,
            "mi_jsd": mi_jsd, "mi_mta": mi_mta}


def logistic_weights(jsd_norm: np.ndarray, one_minus_mta: np.ndarray,
                     y: np.ndarray) -> dict:
    X = np.column_stack([jsd_norm, one_minus_mta])
    lr = LogisticRegression(max_iter=500).fit(X, y)
    coefs = np.abs(lr.coef_[0])
    s = coefs.sum() + 1e-12
    return {"alpha": coefs[0] / s, "beta": coefs[1] / s}
