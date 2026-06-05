"""3-class NLI label space for the ChaosNLI text path (analogue of labels.py).

ChaosNLI label_dist / label_count index order is [entailment, neutral, contradiction]
(verified against the dataset's own `entropy` field). Single-letter answers A/B/C
keep the soft-label-from-logprobs trick: one class == one token across tokenizers.
"""
from __future__ import annotations
import numpy as np

NLI_CLASSES = ["entailment", "neutral", "contradiction"]   # index order MATCHES label_dist
LETTERS = [chr(ord("A") + i) for i in range(len(NLI_CLASSES))]   # A,B,C
LETTER_TO_IDX = {l: i for i, l in enumerate(LETTERS)}
IDX_TO_CLASS = dict(enumerate(NLI_CLASSES))
K = len(NLI_CLASSES)
FLOOR = -20.0

def options_block() -> str:
    return ", ".join(f"{l}={c}" for l, c in zip(LETTERS, NLI_CLASSES))

def logprobs_to_dist(letter_logprobs: dict[str, float], floor: float = FLOOR) -> np.ndarray:
    """{letter: logprob} -> normalized 3-class distribution (sums to 1)."""
    lp = np.full(K, floor, dtype=np.float64)
    for ltr, v in letter_logprobs.items():
        i = LETTER_TO_IDX.get(str(ltr).strip().upper())
        if i is not None:
            lp[i] = max(lp[i], float(v))
    lp -= lp.max()
    p = np.exp(lp)
    p /= p.sum()
    return p

def verify_single_token(tokenizer, letters=LETTERS) -> dict[str, list[int]]:
    return {l: tokenizer.encode(l, add_special_tokens=False) for l in letters}
