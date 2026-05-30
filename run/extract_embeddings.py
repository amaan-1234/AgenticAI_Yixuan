"""Entry point: extract real DINOv2 embeddings for all CIFAR-10H images (Phase 1.4).

    python -m run.extract_embeddings

Verifies the embeddings carry signal: a logistic pre-router trained on them should
predict 'truly hard' images well above chance (ROC-AUC >> 0.5).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from cac.data import cifar10h
from cac.embeddings import dino
from cac.targets import hard_mask


def main():
    images, human_probs, _ = cifar10h.prepare()
    emb = dino.extract_and_cache(images)

    assert emb.shape == (len(images), 384), f"unexpected shape {emb.shape}"
    assert np.isfinite(emb).all(), "embeddings contain NaN/inf"

    truly_hard, _, _ = hard_mask(human_probs)
    idx_tr, idx_va = train_test_split(np.arange(len(emb)), test_size=0.3, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(emb[idx_tr], truly_hard[idx_tr])
    auc = roc_auc_score(truly_hard[idx_va], clf.predict_proba(emb[idx_va])[:, 1])

    print(f"[verify] embeddings {emb.shape}, finite=OK")
    print(f"[verify] pre-router ROC-AUC predicting truly_hard = {auc:.4f} (chance=0.5)")
    assert auc > 0.6, f"embeddings carry little signal (AUC={auc:.3f})"
    print("[ok] DINOv2 embeddings extracted and verified.")


if __name__ == "__main__":
    main()
