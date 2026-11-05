"""Bootstrap confidence intervals: checked against scipy, plus error cases."""

import numpy as np
import pytest
from scipy.stats import bootstrap

from recommender.maths.confidence_intervals import bootstrap_mean_ci, paired_bootstrap_diff_ci


def test_shooting_example_more_shots_means_a_tighter_interval():
    few = bootstrap_mean_ci([1] * 9 + [0] * 11)
    many = bootstrap_mean_ci([1] * 450 + [0] * 550)
    assert few.mean == pytest.approx(0.45) and many.mean == pytest.approx(0.45)
    assert (many.high - many.low) < (few.high - few.low) / 4


@pytest.mark.parametrize("data_seed", [1, 2, 3])
def test_matches_scipy_bootstrap(data_seed):
    values = np.random.default_rng(data_seed).exponential(size=500)
    ours = bootstrap_mean_ci(values, n_resamples=20_000, seed=data_seed)
    theirs = bootstrap((values,), np.mean, n_resamples=20_000, method="percentile", random_state=data_seed)
    # Different random streams, so allow a small gap: 2% of the interval width
    width = ours.high - ours.low
    assert ours.low == pytest.approx(theirs.confidence_interval.low, abs=0.02 * width)
    assert ours.high == pytest.approx(theirs.confidence_interval.high, abs=0.02 * width)


def test_same_seed_same_interval():
    values = np.arange(50.0)
    assert bootstrap_mean_ci(values, seed=7) == bootstrap_mean_ci(values, seed=7)


def test_paired_difference_spots_a_real_win():
    rng = np.random.default_rng(0)
    a = rng.random(300)
    b = a + 0.05  # b is better than a for every single investor
    diff = paired_bootstrap_diff_ci(a, b)
    assert diff.mean == pytest.approx(0.05)
    assert diff.low > 0 and diff.excludes_zero


def test_bad_inputs_fail_loudly():
    with pytest.raises(ValueError, match="empty"):
        bootstrap_mean_ci([])
    with pytest.raises(ValueError, match="NaN"):
        bootstrap_mean_ci([1.0, np.nan])
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_diff_ci([1.0, 2.0], [1.0])
