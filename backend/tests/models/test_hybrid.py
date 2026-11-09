"""Hybrid: the weight really mixes the two models."""

import numpy as np
import pytest

from recommender.models.hybrid import HybridRanker


class Fixed:
    def __init__(self, scores):
        self.scores = np.array(scores, dtype=float)

    def fit(self, train):
        return self

    def score(self, i, investor, candidates):
        return self.scores


CANDIDATES = np.array(["x", "y", "z"])
A, B = Fixed([3, 2, 1]), Fixed([1, 2, 3])  # opposite opinions


def test_weight_one_follows_a_and_zero_follows_b():
    assert np.argmax(HybridRanker(A, B, 1.0).score(0, "i", CANDIDATES)) == 0
    assert np.argmax(HybridRanker(A, B, 0.0).score(0, "i", CANDIDATES)) == 2


def test_scale_does_not_decide_the_winner():
    big_a = Fixed([3000, 2000, 1000])  # same opinion as A, 1000 times bigger
    assert np.allclose(HybridRanker(big_a, B, 0.5).score(0, "i", CANDIDATES), HybridRanker(A, B, 0.5).score(0, "i", CANDIDATES))


def test_bad_weight_fails_loudly():
    with pytest.raises(ValueError, match="between 0 and 1"):
        HybridRanker(A, B, 1.5)
