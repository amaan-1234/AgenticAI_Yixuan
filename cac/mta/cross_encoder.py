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

def _rationale_to_text(s):
    if isinstance(s, dict):
        parts = []
        for k in ("reasoning", "conclusion", "key_features"):
            v = s.get(k)
            if isinstance(v, (list, tuple)):
                v = ", ".join(str(x) for x in v)
            if v:
                parts.append(str(v))
        return ". ".join(parts).strip()
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return s.strip()


MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class MTAScorer:
    """Lazy wrapper around the cross-encoder so the model loads once and is reused."""

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_id, device=device or config.device())

    def pair_scores(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Sigmoid-squashed agreement score in [0,1] for each (text_a, text_b) pair.
        Tokenize manually to bypass CrossEncoder's collator (version-fragile)."""
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        import torch
        def _s(x):
            if isinstance(x, dict):
                x = x.get("reasoning") or x.get("conclusion") or ""
            if isinstance(x, (list, tuple)):
                x = ", ".join(map(str, x))
            x = str(x)
            # strip null bytes + lone surrogates the fast tokenizer rejects
            x = x.replace("\x00", " ")
            x = x.encode("utf-8", "ignore").decode("utf-8", "ignore")
            x = "".join(ch for ch in x if ch.isprintable() or ch.isspace())
            return (x.strip() or "[empty]")
        a = [_s(p_[0]) for p_ in pairs]
        b = [_s(p_[1]) for p_ in pairs]
        model = self.model.model
        tok = self.model.tokenizer
        device = next(model.parameters()).device
        out = []
        bs = 64
        model.eval()
        with torch.no_grad():
            for k in range(0, len(a), bs):
                feats = tok(a[k:k+bs], b[k:k+bs], padding=True, truncation=True,
                            max_length=256, return_tensors="pt").to(device)
                logits = model(**feats).logits.squeeze(-1)
                out.append(logits.float().cpu().numpy().reshape(-1))
        raw = np.asarray(np.concatenate(out), dtype=np.float64)
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
            clean = [t for t in (_rationale_to_text(r) for r in rats) if t]
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
