"""Data alignment gate (requires the cached CIFAR-10H/CIFAR-10 from prepare_data)."""

import numpy as np
import pytest

from cac import config
from cac.data import cifar10h


@pytest.mark.skipif(not config.CIFAR10_LABELS.exists(),
                    reason="run `python -m run.prepare_data` first")
def test_alignment_gate():
    human_probs = cifar10h.download_probs()
    labels = np.load(config.CIFAR10_LABELS)
    rep = cifar10h.alignment_report(human_probs, labels)
    assert rep["n"] == 10000 and rep["k"] == 10
    assert rep["agreement"] >= 0.93, f"alignment {rep['agreement']:.3f} too low"
