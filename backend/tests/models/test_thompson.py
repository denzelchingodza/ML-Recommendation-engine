"""Thompson ranker on a fixed fake model."""

import numpy as np
import pandas as pd
import pytest

from recommender.models.thompson import ThompsonRanker


class Fixed:
    name = "fixed"

    def __init__(self, scores):
        self.scores = np.array(scores, dtype=float)

    def fit(self, train):
        return self

    def score(self, i, investor, candidates):
        return self.scores


CANDIDATES = np.array(["famous", "known", "new_a", "new_b"])
TRAIN = pd.DataFrame(
    {
        "investor_permalink": [f"i{j}" for j in range(30)] + ["i0", "i1"],
        "company_permalink": ["famous"] * 30 + ["known", "known"],
    }
)


def test_pass_mark_1_no_wobble_keeps_the_base_order():
    model = ThompsonRanker(Fixed([4, 3, 2, 1]), 0.0).fit(TRAIN)
    assert list(np.argsort(-model.score(0, "x", CANDIDATES))) == [0, 1, 2, 3]


def test_pass_mark_3_same_seed_same_list_different_seed_different_list():
    a = ThompsonRanker(Fixed([4, 3, 2, 1]), 2.0, seed=1).fit(TRAIN)
    b = ThompsonRanker(Fixed([4, 3, 2, 1]), 2.0, seed=1).fit(TRAIN)
    c = ThompsonRanker(Fixed([4, 3, 2, 1]), 2.0, seed=2).fit(TRAIN)
    assert np.array_equal(a.score(0, "x", CANDIDATES), b.score(0, "x", CANDIDATES))
    assert not np.array_equal(a.score(0, "x", CANDIDATES), c.score(0, "x", CANDIDATES))


def test_new_startups_move_around_much_more_than_famous_ones():
    model = ThompsonRanker(Fixed([0, 0, 0, 0]), 1.0).fit(TRAIN)
    draws = np.array([model.score(i, "x", CANDIDATES) for i in range(2_000)])
    spread = draws.std(axis=0)
    assert spread[0] < spread[1] < spread[2]          # famous (30 deals) < known (2) < new (0)
    assert spread[2] == pytest.approx(spread[3], rel=0.1)


def test_bad_use_fails_loudly():
    with pytest.raises(ValueError, match="negative"):
        ThompsonRanker(Fixed([1]), -0.5)
    with pytest.raises(RuntimeError, match="before fit"):
        ThompsonRanker(Fixed([1, 2, 3, 4]), 1.0).score(0, "x", CANDIDATES)
