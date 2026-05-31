"""ChaosNLI loader — SCAFFOLD for the cross-domain text path (Phase 1.2).

ChaosNLI (Nie et al. 2020, https://github.com/easonnie/ChaosNLI) provides ~100 human
NLI annotations per example, giving a soft label distribution over
{entailment, neutral, contradiction} — the text analogue of CIFAR-10H's human
disagreement. Combined SNLI + MNLI subset ≈ 3,113 (add αNLI for ~4,645, but αNLI is
2-class and handled separately).

This stub defines the intended interface so the text ensemble drops into the same
pipeline (JSD over the 3-class distributions, MTA over rationales, sentence
embeddings for the pre-router). Implement download/parse in Phase 1.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLASSES = ["entailment", "neutral", "contradiction"]
# Manual download: grab the ChaosNLI release from the GitHub repo above and place the
# *.jsonl files under data/chaosnli/. Each line has: uid, example{premise,hypothesis},
# label_count / label_dist (human distribution), majority_label.
SENTENCE_EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # pre-router embeddings


@dataclass
class ChaosNLIItem:
    uid: str
    premise: str
    hypothesis: str
    human_dist: np.ndarray  # shape (3,), over CLASSES


def load(split: str = "snli_mnli") -> list[ChaosNLIItem]:
    raise NotImplementedError(
        "Phase 1.2: parse data/chaosnli/*.jsonl into ChaosNLIItem list. "
        "Mirror the CIFAR-10H flow: VLM->LLM, 10-class->3-class schema, "
        "DINOv2->all-MiniLM-L6-v2 sentence embeddings."
    )


def sentence_embeddings(items: list[ChaosNLIItem]) -> np.ndarray:
    """Pre-router features: embed 'premise [SEP] hypothesis' with all-MiniLM-L6-v2."""
    raise NotImplementedError("Phase 1.2: SentenceTransformer(SENTENCE_EMB_MODEL).encode(...)")
