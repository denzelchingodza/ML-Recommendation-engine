"""Cluster bootstrap: resample whole investors, not single rows. Written BEFORE the code."""

import numpy as np
import pytest

from recommender.maths.confidence_intervals import bootstrap_mean_ci, cluster_bootstrap_mean_ci


def test_one_row_per_cluster_is_the_plain_bootstrap():
    # Every player plays one game: picking players IS picking games, so nothing changes.
    values = np.random.default_rng(0).normal(size=300)
    clusters = np.arange(300)
    assert cluster_bootstrap_mean_ci(values, clusters, seed=3) == bootstrap_mean_ci(values, seed=3)


def test_copying_every_row_does_not_fake_a_tighter_interval():
    # Same player listed twice is not new evidence. The plain bootstrap gets fooled, the cluster one does not.
    values = np.random.default_rng(1).normal(size=200)
    clusters = np.arange(200)
    once = cluster_bootstrap_mean_ci(values, clusters, seed=5)
    twice = cluster_bootstrap_mean_ci(np.repeat(values, 2), np.repeat(clusters, 2), seed=5)
    naive_twice = bootstrap_mean_ci(np.repeat(values, 2), seed=5)
    assert twice.mean == pytest.approx(once.mean)
    assert (twice.high - twice.low) == pytest.approx(once.high - once.low, rel=1e-9)
    assert (naive_twice.high - naive_twice.low) < 0.8 * (once.high - once.low)


def test_mean_weights_every_row_equally():
    ci = cluster_bootstrap_mean_ci([1.0, 1.0, 1.0, 0.0], ["a", "a", "a", "b"])
    assert ci.mean == pytest.approx(0.75)
    assert ci.n == 4


def test_same_seed_same_interval():
    values, clusters = np.arange(40.0), np.arange(40) // 4
    assert cluster_bootstrap_mean_ci(values, clusters, seed=2) == cluster_bootstrap_mean_ci(values, clusters, seed=2)


def test_lengths_must_match():
    with pytest.raises(ValueError, match="same length"):
        cluster_bootstrap_mean_ci([1.0, 2.0], ["a"])


def test_nan_is_refused():
    with pytest.raises(ValueError, match="NaN"):
        cluster_bootstrap_mean_ci([1.0, np.nan], ["a", "b"])
