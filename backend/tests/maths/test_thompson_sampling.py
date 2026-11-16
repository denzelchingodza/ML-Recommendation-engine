"""Thompson sampling maths: the wobble shrinks with evidence, and has the right size."""

import numpy as np
import pytest

from recommender.maths.thompson_sampling import exploration_spread, sample_scores


def test_pass_mark_2_more_deals_means_a_smaller_wobble():
    spread = exploration_spread([0, 3, 99], strength=1.0)
    assert spread == pytest.approx([1.0, 0.5, 0.1])


def test_strength_zero_means_no_wobble_at_all():
    scores = np.array([0.3, -1.2, 2.0])
    assert np.array_equal(sample_scores(scores, [0, 0, 5], 0.0, np.random.default_rng(1)), scores)


def test_wobble_really_has_the_promised_size():
    rng = np.random.default_rng(0)
    draws = sample_scores(np.zeros(200_000), np.zeros(200_000), 2.0, rng)
    assert draws.std() == pytest.approx(2.0, rel=0.01)
    assert abs(draws.mean()) < 0.02


def test_bad_inputs_fail_loudly():
    with pytest.raises(ValueError, match="negative"):
        exploration_spread([0], -1.0)
    with pytest.raises(ValueError, match="negative"):
        exploration_spread([-1], 1.0)
    with pytest.raises(ValueError, match="same shape"):
        sample_scores([1.0, 2.0], [0], 1.0, np.random.default_rng(0))
