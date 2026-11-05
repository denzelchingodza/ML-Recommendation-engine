"""recall@k and NDCG@k: hand examples, error cases, and pass mark 1 (match scikit-learn)."""

import numpy as np
import pytest
from sklearn.metrics import ndcg_score, recall_score

from recommender.maths.ranking_metrics import ndcg_at_k, recall_at_k

HIDDEN = {"A", "C"}


@pytest.mark.parametrize("k, expected", [(1, 0.0), (3, 0.5), (5, 1.0)])
def test_recall_hand_example(k, expected):
    assert recall_at_k(["B", "A", "D", "E", "C"], HIDDEN, k) == expected


def test_ndcg_rewards_hits_near_the_top():
    top = ndcg_at_k(["A", "C", "B", "D", "E"], HIDDEN, 5)
    bottom = ndcg_at_k(["B", "D", "E", "A", "C"], HIDDEN, 5)
    assert top == pytest.approx(1.0)
    assert bottom == pytest.approx((1 / np.log2(5) + 1 / np.log2(6)) / (1 + 1 / np.log2(3)))  # about 0.50


def test_ndcg_ideal_is_capped_at_k():
    # 3 hidden deals, k = 2, both top spots are hits: that is the best possible, so 1.0
    assert ndcg_at_k(["A", "C", "E"], {"A", "C", "E"}, 2) == pytest.approx(1.0)


def test_k_longer_than_the_list_is_fine():
    assert recall_at_k(["A", "B"], {"A"}, 10) == 1.0


@pytest.mark.parametrize("func", [recall_at_k, ndcg_at_k])
def test_bad_inputs_fail_loudly(func):
    with pytest.raises(ValueError, match="k must be at least 1"):
        func(["A"], {"A"}, 0)
    with pytest.raises(ValueError, match="nothing to find"):
        func(["A"], set(), 1)


def test_pass_mark_1_matches_scikit_learn_on_1000_random_cases():
    rng = np.random.default_rng(123)
    for _ in range(1_000):
        n = int(rng.integers(2, 60))
        k = int(rng.integers(1, n + 5))
        companies = np.array([f"c{i}" for i in range(n)])
        n_relevant = int(rng.integers(1, n + 1))
        relevant = set(rng.choice(companies, size=n_relevant, replace=False))
        ranked = rng.permutation(companies)

        y_true = np.array([1 if c in relevant else 0 for c in ranked])
        # scores that reproduce our ranking exactly (no ties): first place gets the highest score
        y_score = np.arange(n, 0, -1)
        in_top_k = (np.arange(n) < k).astype(int)

        assert recall_at_k(ranked, relevant, k) == pytest.approx(recall_score(y_true, in_top_k), abs=1e-12)
        assert ndcg_at_k(ranked, relevant, k) == pytest.approx(ndcg_score([y_true], [y_score], k=k), abs=1e-12)
