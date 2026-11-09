"""Pairwise ranking loss: checked against scipy and a numerical slope."""

import numpy as np
import pytest
from scipy.special import expit

from recommender.maths.pairwise_ranking_loss import bpr_loss, bpr_loss_slope, sigmoid


def test_sigmoid_matches_scipy_including_extremes():
    x = np.array([-1000.0, -20.0, -1.0, 0.0, 1.0, 20.0, 1000.0])
    assert sigmoid(x) == pytest.approx(expit(x), abs=1e-15)


def test_loss_values_in_plain_words():
    assert bpr_loss(0.0) == pytest.approx(np.log(2))  # coin flip
    assert bpr_loss(50.0) < 1e-20                    # sure and right: no loss
    assert bpr_loss(-50.0) == pytest.approx(50.0)    # sure and wrong: big loss, no overflow


def test_slope_matches_a_numerical_slope():
    x = np.linspace(-8, 8, 101)
    eps = 1e-6
    numerical = (bpr_loss(x + eps) - bpr_loss(x - eps)) / (2 * eps)
    assert bpr_loss_slope(x) == pytest.approx(numerical, abs=1e-8)
