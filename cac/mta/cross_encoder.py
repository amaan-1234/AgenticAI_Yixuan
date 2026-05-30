"""Macro Trace Agreement (MTA) via a cross-encoder (Phase 1.3).

Replaces v4's `simulate_mta` (a noisy function of human entropy) with real pairwise
semantic scoring of the model rationales.

    MTA_i = (2 / M(M-1)) * sum_{a<b} sigmoid(CE(R_ia, R_ib))

`ms-marco-MiniLM-L-6-v2` (per the lit review) is a relevance cross-encoder, so raw
outputs are logits; we squash with a sigmoid to land in [0,1]. The escalation signal
uses (1 - MTA): low rationale agreement -> high escalation, even when soft-label JSD
is low (the case where models agree on the label but disagree on the reasoning).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from cac import config

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class MTAScorer:
    """Lazy wrapper around the cross-encoder so the model loads once and is reused."""

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_id, device=device or config.device())

    def pair_scores(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Sigmoid-squashed agreement score in [0,1] for each (text_a, text_b) pair."""
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        raw = np.asarray(self.model.predict(pairs), dtype=np.float64)
        return _sigmoid(raw).astype(np.float32)

    def mta_for_rationales(self, rationales_per_item: list[list[str]]) -> np.ndarray:
        """Mean pairwise agreement per item. rationales_per_item: [N][M] strings.

        Items with <2 non-empty rationales get MTA=0 (treated as max disagreement,
        so they escalate — the safe default when reasoning can't be compared).
        """
        # Flatten all pairs across items, score in one batch, then regroup.
        flat_pairs: list[tuple[str, str]] = []
        spans: list[tuple[int, int]] = []
        for rats in rationales_per_item:
            clean = [r for r in rats if r and r.strip()]
            start = len(flat_pairs)
            flat_pairs.extend(combinations(clean, 2))
            spans.append((start, len(flat_pairs)))

        scores = self.pair_scores(flat_pairs)
        mta = np.zeros(len(rationales_per_item), dtype=np.float32)
        for i, (s, e) in enumerate(spans):
            mta[i] = float(scores[s:e].mean()) if e > s else 0.0
        return mta


def compute_and_cache(rationales_per_item: list[list[str]], out_path=config.MTA_SCORES,
                      model_id: str = MODEL_ID) -> np.ndarray:
    """Compute MTA for every item and cache to .npy."""
    scorer = MTAScorer(model_id)
    mta = scorer.mta_for_rationales(rationales_per_item)
    np.save(out_path, mta)
    print(f"[mta] saved {mta.shape} -> {out_path}")
    return mta
