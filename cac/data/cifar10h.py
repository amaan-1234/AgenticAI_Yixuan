"""CIFAR-10H data: human label distributions + aligned CIFAR-10 test images.

`cifar10h-probs.npy` holds (10000, 10) crowd label distributions ordered to match
the *standard CIFAR-10 test set*. We download the matching test images via
torchvision (same order) and assert agreement between argmax(human_probs) and the
CIFAR-10 ground-truth labels before anything downstream runs — a silent ordering
mismatch would invalidate every metric.
"""

from __future__ import annotations

import urllib.request

import numpy as np

from cac import config
from cac.data.labels import CIFAR10_CLASSES

PROBS_URL = (
    "https://raw.githubusercontent.com/jcpeterson/cifar-10h/master/data/"
    "cifar10h-probs.npy"
)


def download_probs(path=config.CIFAR10H_PROBS) -> np.ndarray:
    """Download (if needed) and return normalized human label distributions."""
    if not path.exists():
        print(f"[data] downloading cifar10h-probs.npy -> {path}")
        urllib.request.urlretrieve(PROBS_URL, path)
    probs = np.load(path).astype(np.float64)
    probs = np.clip(probs, 1e-12, None)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def download_cifar10_test():
    """Return (images uint8 [N,32,32,3], labels int [N]) in standard test order."""
    if config.CIFAR10_IMAGES.exists() and config.CIFAR10_LABELS.exists():
        return np.load(config.CIFAR10_IMAGES), np.load(config.CIFAR10_LABELS)

    from torchvision.datasets import CIFAR10

    print(f"[data] downloading CIFAR-10 test set -> {config.DATA_DIR}")
    ds = CIFAR10(root=str(config.DATA_DIR), train=False, download=True)
    images = ds.data.astype(np.uint8)            # (10000, 32, 32, 3), standard order
    labels = np.array(ds.targets, dtype=np.int64)
    np.save(config.CIFAR10_IMAGES, images)
    np.save(config.CIFAR10_LABELS, labels)
    return images, labels


def alignment_report(human_probs: np.ndarray, labels: np.ndarray) -> dict:
    """Compare argmax(human_probs) to CIFAR-10 labels. Agreement should be ~0.95."""
    assert human_probs.shape == (len(labels), len(CIFAR10_CLASSES)), (
        f"shape mismatch: probs {human_probs.shape} vs labels {labels.shape}"
    )
    agree = float((human_probs.argmax(axis=1) == labels).mean())
    return {"n": len(labels), "k": human_probs.shape[1], "agreement": agree}


def prepare(min_agreement: float = 0.93):
    """Download everything, run the alignment gate, return (images, human_probs, labels).

    Raises AssertionError if argmax(human_probs) vs CIFAR-10 labels agreement is
    below `min_agreement` (CIFAR-10H reports ~95%; 0.93 leaves headroom for noise).
    """
    human_probs = download_probs()
    images, labels = download_cifar10_test()
    rep = alignment_report(human_probs, labels)
    print(
        f"[data] N={rep['n']} K={rep['k']} | "
        f"human-argmax vs CIFAR-10 labels agreement = {rep['agreement']:.4f}"
    )
    assert rep["agreement"] >= min_agreement, (
        f"ALIGNMENT GATE FAILED: agreement {rep['agreement']:.4f} < {min_agreement}. "
        "Image/label ordering likely does not match cifar10h-probs.npy."
    )
    return images, human_probs, labels
