"""Dataset-agnostic source for human_probs + per-item embeddings.
Lets report.py / run_pipeline.py work on CIFAR-10H or ChaosNLI via one flag.
"""
from __future__ import annotations
import numpy as np
from cac import config

def human_probs(dataset: str) -> np.ndarray:
    if dataset == "cifar10h":
        from cac.data import cifar10h
        _, hp, _ = cifar10h.prepare()
        return hp
    if dataset == "chaosnli":
        from cac.data import chaosnli
        items = chaosnli.load("snli_mnli")
        return np.vstack([it.human_dist for it in items])
    raise SystemExit(f"unknown dataset {dataset}")

def embeddings(dataset: str, n: int) -> np.ndarray:
    if dataset == "cifar10h":
        return np.load(config.DINO_EMB)[:n]
    if dataset == "chaosnli":
        path = config.OUTPUTS_DIR / "chaosnli_sent_emb.npy"
        if path.exists():
            emb = np.load(path)
        else:
            from cac.data import chaosnli
            items = chaosnli.load("snli_mnli")
            emb = chaosnli.sentence_embeddings(items)
            np.save(path, emb)
            print(f"[emb] cached {emb.shape} -> {path}")
        return emb[:n]
    raise SystemExit(f"unknown dataset {dataset}")
