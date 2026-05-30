"""Calibration of the escalation signal -> P(truly hard): Platt vs Isotonic (v4 section 7).

ECE and budget-constrained threshold selection are unchanged from v4. Isotonic is
preferred at CIFAR-10H scale (>=500 calibration images); Platt for smaller sets.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def compute_ece(p_pred: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    """Expected Calibration Error + per-bin accuracies/confidences."""
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs, bin_confs = [], []
    for b in range(n_bins):
        mask = (p_pred >= edges[b]) & (p_pred < edges[b + 1])
        if mask.sum() == 0:
            bin_accs.append(0); bin_confs.append(0)
            continue
        acc = y_true[mask].mean()
        conf = p_pred[mask].mean()
        ece += mask.sum() * abs(acc - conf)
        bin_accs.append(acc); bin_confs.append(conf)
    return ece / len(y_true), bin_accs, bin_confs


def best_threshold(p_val: np.ndarray, y_val: np.ndarray, budget: float) -> dict:
    """Threshold maximizing recall subject to escalation rate <= budget."""
    best = {"tau": 0.0, "prec": 0.0, "rec": 0.0, "cost": 1.0}
    for tau in np.linspace(0, 1, 300):
        esc = p_val >= tau
        tp = np.sum(esc & (y_val == 1)); fp = np.sum(esc & (y_val == 0))
        fn = np.sum(~esc & (y_val == 1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        cost = esc.mean()
        if cost <= budget and rec > best["rec"]:
            best = {"tau": float(tau), "prec": float(prec), "rec": float(rec),
                    "cost": float(cost)}
    return best


def fit_calibrators(sig: np.ndarray, y_all: np.ndarray, idx_train, idx_val,
                    budget: float) -> dict:
    """Fit Platt + Isotonic on the raw signal, pick the lower-ECE calibrator."""
    y_train, y_val = y_all[idx_train].astype(float), y_all[idx_val].astype(float)

    platt = LogisticRegression().fit(sig[idx_train].reshape(-1, 1), y_train)
    p_platt = platt.predict_proba(sig[idx_val].reshape(-1, 1))[:, 1]
    ece_platt, ba_p, bc_p = compute_ece(p_platt, y_val)
    thr_platt = best_threshold(p_platt, y_val, budget)

    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(sig[idx_train], y_train)
    p_iso = iso.predict(sig[idx_val])
    ece_iso, ba_i, bc_i = compute_ece(p_iso, y_val)
    thr_iso = best_threshold(p_iso, y_val, budget)

    # Callables to apply each (train-fitted) calibrator to ANY signal array — lets the
    # results DB compute P(hard) for all N images, not just the val split.
    def predict_platt(x):
        return platt.predict_proba(np.asarray(x).reshape(-1, 1))[:, 1]

    def predict_iso(x):
        return iso.predict(np.asarray(x))

    if ece_iso <= ece_platt:
        best = {"name": "Isotonic", "p_val": p_iso, "ece": ece_iso, "predict": predict_iso,
                **thr_iso}
    else:
        best = {"name": "Platt", "p_val": p_platt, "ece": ece_platt, "predict": predict_platt,
                **thr_platt}

    return {
        "platt": {"p_val": p_platt, "ece": ece_platt, "bin_acc": ba_p,
                  "predict": predict_platt, **thr_platt},
        "iso": {"p_val": p_iso, "ece": ece_iso, "bin_acc": ba_i,
                "predict": predict_iso, **thr_iso},
        "best": best, "y_val": y_val,
    }
