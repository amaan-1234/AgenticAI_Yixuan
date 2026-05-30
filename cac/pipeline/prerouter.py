"""Pre-router: skip trivially-easy images before the ensemble (v4 section 8).

Logistic head on frozen embeddings (real DINOv2 in --real, synthetic CLIP in
--simulate) with an asymmetric loss: false-easies (a hard image wrongly skipped,
bypassing both ensemble AND frontier) are penalized `penalty`x. Logic unchanged
from v4; only the embedding source differs.

INVESTIGATION NOTE (2026-05, WSL2 N=100 smoke):
    Measured false-easy rate = 22.2% — well above the v4 simulation's ~0.1% and
    the 8x asymmetric-loss design target. Three plausible causes, ranked:
      1. Small-sample artifact on N=100 (very likely): the train/val split is
         70 train / 30 val and `truly_hard` covers only the top-30% by entropy,
         so the validation false-easy count divides by ~9 hard images — a single
         misroute shows as ~11%.
      2. PREROUTE_THRESHOLD = 0.15 too aggressive on real DINOv2 features (the
         synthetic CLIP embeddings in v4 separated easy/hard far more cleanly).
      3. 8x asymmetric weight insufficient when easy-class density is high.

    Decision: re-measure at N=10000 on the HPC pilot before tuning. If the rate
    stays > ~5% at N=10k, sweep PREROUTE_THRESHOLD in {0.05, 0.10, 0.15, 0.20}
    and `penalty` in {8, 12, 16, 24} and pick the threshold that drives false-easy
    < 1% at the highest skip rate. Threshold sweep is already on the Phase-3.2
    roadmap (sensitivity analysis), so this folds in cleanly.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def train_prerouter(embeddings: np.ndarray, truly_hard: np.ndarray, idx_train, idx_val,
                    threshold: float = 0.15, penalty: float = 8.0) -> dict:
    sw = np.ones(len(idx_train))
    sw[truly_hard[idx_train]] = penalty
    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(embeddings[idx_train], truly_hard[idx_train].astype(int), sample_weight=sw)

    proba = clf.predict_proba(embeddings[idx_val])[:, 1]
    easy = proba < threshold
    false_easy = int(np.sum(easy & truly_hard[idx_val]))
    n_hard_val = int(truly_hard[idx_val].sum())
    return {
        "proba": proba, "easy": easy, "hard": ~easy,
        "n_skipped": int(easy.sum()),
        "skip_rate": float(easy.mean()),
        "false_easy": false_easy,
        "false_easy_rate": false_easy / max(n_hard_val, 1),
        "threshold": threshold,
    }
