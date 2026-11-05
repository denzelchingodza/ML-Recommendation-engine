"""The two starters on tiny hand-made data."""

import numpy as np
import pandas as pd
import pytest

from recommender.models.baselines import MostPopularRanker, RandomRanker

TRAIN = pd.DataFrame(
    {
        "investor_permalink": ["i1", "i2", "i3", "i1", "i1"],
        "company_permalink": ["pop", "pop", "pop", "mid", "mid"],  # i1 backed "mid" twice
    }
)


def test_popularity_counts_different_investors_not_rows():
    model = MostPopularRanker().fit(TRAIN)
    scores = model.score(0, "x", np.array(["pop", "mid", "brand_new"]))
    assert list(scores) == [3.0, 1.0, 0.0]  # "mid" counts once; unseen companies score 0


def test_most_popular_refuses_to_score_before_fit():
    with pytest.raises(RuntimeError, match="before fit"):
        MostPopularRanker().score(0, "x", np.array(["pop"]))


def test_random_is_repeatable_and_differs_per_investor():
    model = RandomRanker(seed=5)
    candidates = np.array(["a", "b", "c", "d"])
    assert np.array_equal(model.score(0, "x", candidates), model.score(0, "x", candidates))
    assert not np.array_equal(model.score(0, "x", candidates), model.score(1, "y", candidates))
