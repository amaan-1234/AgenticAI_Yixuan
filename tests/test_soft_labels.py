"""Soft-label extraction from token logprobs (no GPU)."""

import numpy as np

from cac.data.labels import LETTERS
from cac.ensemble.soft_labels import logprobs_to_dist


def test_sums_to_one_and_argmax():
    # log-probs: B highest, A second.
    lp = {"A": -1.0, "B": -0.2, "C": -3.0}
    d = logprobs_to_dist(lp)
    assert d.shape == (10,)
    assert abs(d.sum() - 1.0) < 1e-9
    assert int(d.argmax()) == LETTERS.index("B")


def test_missing_letters_get_floor():
    lp = {"A": 0.0}  # only one class observed in top-k
    d = logprobs_to_dist(lp)
    assert d.argmax() == 0
    # the 9 unobserved classes share the floor mass, each tiny but > 0
    assert (d[1:] > 0).all()
    assert d[0] > d[1] * 100  # observed class dominates


def test_case_and_whitespace_robust():
    d1 = logprobs_to_dist({"a": -0.1, "B": -2.0})
    d2 = logprobs_to_dist({" A ": -0.1, "b": -2.0})
    assert np.allclose(d1, d2)
