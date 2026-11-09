"""Cosine similarity: hand examples, zero vectors, and pass mark 1 (match scikit-learn)."""

import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from recommender.maths.cosine_similarity import cosine_similarity, cosine_similarity_to_rows


def test_hand_examples():
    software_ny = [1, 0, 0, 0, 1]
    assert cosine_similarity(software_ny, software_ny) == pytest.approx(1.0)        # identical
    assert cosine_similarity(software_ny, [0, 1, 0, 1, 0]) == pytest.approx(0.0)    # nothing shared
    assert cosine_similarity(software_ny, [1, 0, 0, 1, 0]) == pytest.approx(0.5)    # shares 1 of 2 ticks


def test_only_the_mix_matters_not_the_size():
    small = np.array([2, 0, 1])
    big = small * 20  # same mix, 20 times the deals
    assert cosine_similarity(small, big) == pytest.approx(1.0)


def test_pass_mark_3_empty_checklist_scores_zero_and_never_crashes():
    scores = cosine_similarity_to_rows([1, 0, 1], [[0, 0, 0], [1, 0, 1]])
    assert list(scores) == pytest.approx([0.0, 1.0])
    assert cosine_similarity([0, 0, 0], [1, 1, 1]) == 0.0


def test_bad_shapes_fail_loudly():
    with pytest.raises(ValueError, match="rows must have shape"):
        cosine_similarity_to_rows([1, 0], [[1, 0, 0]])
    with pytest.raises(ValueError, match="NaN"):
        cosine_similarity_to_rows([1, np.nan], [[1, 0]])


def test_pass_mark_1_matches_scikit_learn_on_1000_random_cases():
    rng = np.random.default_rng(7)
    for _ in range(1_000):
        dims = int(rng.integers(1, 30))
        query = rng.integers(0, 5, size=dims).astype(float)
        if not query.any():
            query[0] = 1.0  # scikit-learn and we agree on non-zero vectors; zero vectors are tested above
        rows = rng.integers(0, 5, size=(int(rng.integers(1, 20)), dims)).astype(float)
        rows[~rows.any(axis=1), 0] = 1.0
        ours = cosine_similarity_to_rows(query, rows)
        theirs = sk_cosine(query[None, :], rows)[0]
        assert ours == pytest.approx(theirs, abs=1e-12)
