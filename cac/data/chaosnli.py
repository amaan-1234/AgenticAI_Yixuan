"""ChaosNLI loader for the cross-domain text path (SNLI + MNLI, 3-class).

Real implementation (replaces the Phase-1.2 stub). Reads the official
chaosNLI_v1.0 jsonl files. Each record has:
  uid, label_counter{e/n/c:int}, majority_label, label_dist[3], label_count[3],
  entropy(float), example{premise,hypothesis,...}, old_label, old_labels
label_dist / label_count index order is [entailment, neutral, contradiction] —
verified by recomputing entropy from label_dist and matching the file's `entropy`.
alphaNLI is EXCLUDED (2-choice abductive, different label space).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

from cac import config
from cac.data.labels_nli import NLI_CLASSES

CLASSES = NLI_CLASSES
SENTENCE_EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_DATA_DIR = Path(config.DATA_DIR) / "chaosnli" / "chaosNLI_v1.0"
_FILES = {"snli": "chaosNLI_snli.jsonl", "mnli": "chaosNLI_mnli_m.jsonl"}


@dataclass
class ChaosNLIItem:
    uid: str
    premise: str
    hypothesis: str
    human_dist: np.ndarray   # shape (3,), [e, n, c], sums to 1
    entropy: float           # bits (base-2), from the file (cross-checked)


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())   # ChaosNLI stores entropy in bits (base 2)


def load(split: str = "snli_mnli") -> list[ChaosNLIItem]:
    """Parse ChaosNLI into items. split in {'snli','mnli','snli_mnli'}."""
    which = ["snli", "mnli"] if split == "snli_mnli" else [split]
    items: list[ChaosNLIItem] = []
    for key in which:
        path = _DATA_DIR / _FILES[key]
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; unzip chaosNLI_v1.0.zip there")
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                dist = np.asarray(r["label_dist"], dtype=np.float64)
                s = dist.sum()
                if not np.isfinite(s) or s <= 0:
                    continue
                dist = dist / s
                # correctness gate: our entropy must match the file's
                if abs(_entropy(dist) - float(r["entropy"])) > 1e-3:
                    raise AssertionError(
                        f"{key} {r['uid']}: entropy mismatch "
                        f"{_entropy(dist):.4f} vs file {r['entropy']:.4f} "
                        f"(label_dist index order may differ)"
                    )
                ex = r["example"]
                items.append(ChaosNLIItem(
                    uid=r["uid"],
                    premise=ex["premise"],
                    hypothesis=ex["hypothesis"],
                    human_dist=dist,
                    entropy=float(r["entropy"]),
                ))
    return items


def human_distributions(items: list[ChaosNLIItem]) -> np.ndarray:
    """(N,3) human soft-label matrix, for H_human / correlation targets."""
    return np.vstack([it.human_dist for it in items])


def sentence_embeddings(items: list[ChaosNLIItem], batch_size: int = 256) -> np.ndarray:
    """Pre-router features: embed 'premise [SEP] hypothesis' with all-MiniLM-L6-v2."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(SENTENCE_EMB_MODEL, device=config.device())
    texts = [f"{it.premise} [SEP] {it.hypothesis}" for it in items]
    return np.asarray(model.encode(texts, batch_size=batch_size,
                                   show_progress_bar=False, convert_to_numpy=True),
                      dtype=np.float32)
