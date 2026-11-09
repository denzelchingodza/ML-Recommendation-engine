"""Scoring one model on one dataset. Shared by the scoreboard and by tuning.

Kept separate so tuning and the scoreboard do not import each other.
"""

import numpy as np
import pandas as pd

from recommender.evaluation.draft_board import check_board_covers_test
from recommender.maths.ranking_metrics import ndcg_at_k, recall_at_k

DEFAULT_K = 10


def rank_candidates(candidates: np.ndarray, scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Highest score first. Ties are broken RANDOMLY, never alphabetically."""
    tiebreak = rng.random(len(candidates))
    order = np.lexsort((tiebreak, -scores))  # last key sorts first: score, then tiebreak
    return candidates[order]


def score_model(model, dataset, board: np.ndarray, k: int = DEFAULT_K, seed: int = 0) -> pd.DataFrame:
    """One row per test investor, with recall@k and NDCG@k for each segment.

    Segments: overall (all hidden deals), known (companies seen in training),
    new (companies never seen in training). A segment is NaN when the investor
    has no hidden deals of that kind, so it is left out, never counted as 0.
    """
    check_board_covers_test(board, dataset.test)
    train, test = dataset.train, dataset.test
    known_companies = set(train["company_permalink"])
    seen_by_investor = train.groupby("investor_permalink")["company_permalink"].apply(set)
    hidden_by_investor = test.groupby("investor_permalink")["company_permalink"].apply(set)

    rows = []
    for index, investor in enumerate(sorted(hidden_by_investor.index)):
        seen = seen_by_investor.get(investor, set())
        candidates = board[~np.isin(board, list(seen))]  # never recommend what they already backed
        scores = model.score(index, investor, candidates)
        if len(scores) != len(candidates):
            raise ValueError(f"{model.name} returned {len(scores)} scores for {len(candidates)} candidates")
        ranked = rank_candidates(candidates, scores, np.random.default_rng([seed, index]))

        hidden = hidden_by_investor[investor]
        segment_sets = {
            "overall": hidden,
            "known": hidden & known_companies,
            "new": hidden - known_companies,
        }
        row = {"investor": investor, "n_candidates": len(candidates)}
        for segment, relevant in segment_sets.items():
            row[f"n_hidden_{segment}"] = len(relevant)
            for metric, func in (("recall", recall_at_k), ("ndcg", ndcg_at_k)):
                row[f"{metric}_{segment}"] = func(ranked, relevant, k) if relevant else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
