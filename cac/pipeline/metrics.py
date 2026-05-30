"""Correlations, pipeline cost model, and cascade accuracy (v4 sections 9-10)."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def normalise_01(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def correlations(signal: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """(Pearson r, Spearman rho) between a signal and the target (human entropy)."""
    r = float(np.corrcoef(signal, target)[0, 1])
    rho = float(spearmanr(signal, target).statistic)
    return r, rho


def cost_stages(prerouter_hard: np.ndarray, p_val: np.ndarray, tau: float,
                n_val: int) -> dict:
    """Three-stage routing counts on the validation split + cost vs frontier-only."""
    escalated = p_val[prerouter_hard] >= tau
    stage_1 = int((~prerouter_hard).sum())          # pre-router skip
    stage_3 = int(escalated.sum())                   # frontier escalation
    stage_2 = int(prerouter_hard.sum() - stage_3)    # ensemble consensus
    cost_ratio = n_val / stage_3 if stage_3 > 0 else float("inf")
    return {"stage_1": stage_1, "stage_2": stage_2, "stage_3": stage_3,
            "cost_ratio": cost_ratio, "frontier_pct": stage_3 / n_val}


def cascade_accuracy(ensemble_majority: np.ndarray, human_argmax: np.ndarray,
                     escalated: np.ndarray, frontier_acc: float) -> dict:
    """Cascade = ensemble handles non-escalated, frontier handles escalated."""
    easy = ~escalated
    n = len(human_argmax)
    ens_acc_easy = float((ensemble_majority[easy] == human_argmax[easy]).mean()) if easy.any() else 0.0
    acc = (ens_acc_easy * easy.sum() + frontier_acc * escalated.sum()) / n
    return {"cascade_acc": float(acc), "ensemble_acc_easy": ens_acc_easy}
