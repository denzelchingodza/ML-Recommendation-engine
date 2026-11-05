"""Ranking metrics: recall@k and NDCG@k.

Both take one investor's ranked list (best first) and the set of companies
they really backed (the hidden deals), and return a score from 0 to 1.

Checked against scikit-learn in tests/maths/test_ranking_metrics.py.
"""

import numpy as np


def _top_k_hits(ranked_ids, relevant_ids, k: int) -> np.ndarray:
    """Return 1.0 for each of the top k spots holding a hidden deal, else 0.0."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if len(relevant_ids) == 0:
        raise ValueError("relevant_ids is empty: there is nothing to find, so the score is undefined")
    top_k = np.asarray(ranked_ids)[:k]
    return np.isin(top_k, list(relevant_ids)).astype(float)


def recall_at_k(ranked_ids, relevant_ids, k: int) -> float:
    """Share of the hidden deals that appear in the top k.

        recall@k = (hidden deals in the top k) / (total hidden deals)

    Divides by the number of HIDDEN deals, not by k. Dividing by k would be
    precision@k, a different metric.
    """
    hits = _top_k_hits(ranked_ids, relevant_ids, k)
    return float(hits.sum() / len(relevant_ids))


def ndcg_at_k(ranked_ids, relevant_ids, k: int) -> float:
    """Normalised discounted cumulative gain at k.

    Hits near the top count more:
        value of position i (1-based) = 1 / log2(i + 1)  ->  1.00, 0.63, 0.50, 0.43, ...
        DCG       = sum of the values of positions holding a hidden deal
        ideal DCG = DCG if every hidden deal sat at the very top
        NDCG      = DCG / ideal DCG, between 0 and 1

    The ideal uses min(number of hidden deals, k): with 8 hidden deals and
    k = 5, the best possible is 5 hits in the top 5. Without this, nobody
    could ever score 1.0.
    """
    hits = _top_k_hits(ranked_ids, relevant_ids, k)
    positions = np.arange(1, k + 1)
    discounts = 1.0 / np.log2(positions + 1)

    dcg = float(np.sum(hits * discounts[: len(hits)]))
    n_ideal = min(len(relevant_ids), k)
    ideal_dcg = float(np.sum(discounts[:n_ideal]))
    return dcg / ideal_dcg
