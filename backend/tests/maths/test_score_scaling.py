"""Standardising scores: checked against scipy."""

import numpy as np
import pytest
from scipy.stats import zscore

from recommender.maths.score_scaling import standardise


def test_matches_scipy_zscore():
    x = np.random.default_rng(1).exponential(size=200)
    assert standardise(x) == pytest.approx(zscore(x), abs=1e-12)


def test_equal_scores_have_nothing_to_rank():
    assert list(standardise([3.0, 3.0, 3.0])) == [0.0, 0.0, 0.0]


def test_nan_fails_loudly():
    with pytest.raises(ValueError, match="NaN"):
        standardise([1.0, np.nan])
