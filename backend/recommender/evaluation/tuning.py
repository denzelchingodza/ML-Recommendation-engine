"""Pick model settings on a PRACTICE split inside the training data.

The hidden test (deals from 2012) must never influence any choice, or the
final score would be quietly inflated. So we split the training years again:
    practice-train: deals before 1 Jan 2011
    practice-test:  deals from 1 Jan 2011 up to the real cutoff
and choose the settings that score best there.

The grid was written down before the first run.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from recommender.data.split_data import temporal_split
from recommender.evaluation.scoreboard_core import exposure, score_model
from recommender.models.hybrid import HybridRanker
from recommender.models.label_similarity import LabelSimilarityRanker
from recommender.models.matrix_factorisation import MatrixFactorisationRanker
from recommender.models.thompson import ThompsonRanker

PRACTICE_CUTOFF = "2011-01-01"
GRID = tuple({"dim": d, "reg": r} for d in (16, 32) for r in (0.01, 0.001))
BLEND_WEIGHTS = tuple(round(w / 10, 1) for w in range(11))  # 0.0, 0.1, ... 1.0
EXPLORATION_STRENGTHS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)


class TuningError(Exception):
    """Raised when tuning could see data it must not see."""


def practice_split(train: pd.DataFrame, must_end_before: str):
    if (train["first_invested_at"] >= pd.Timestamp(must_end_before)).any():
        raise TuningError(f"Tuning data contains deals on or after {must_end_before}: that is the hidden test")
    p_train, p_test, _ = temporal_split(train, PRACTICE_CUTOFF, min_history=5)
    board = np.array(sorted(set(train.loc[train["first_invested_at"] >= pd.Timestamp(PRACTICE_CUTOFF), "company_permalink"])))
    return SimpleNamespace(train=p_train, test=p_test), board


def select_settings(train: pd.DataFrame, companies: pd.DataFrame, must_end_before: str,
                    grid=GRID, k: int = 10, seed: int = 0):
    """Return the best settings by mean practice NDCG@k, plus every setting's score."""
    practice, board = practice_split(train, must_end_before)
    results = []
    for settings in grid:
        model = MatrixFactorisationRanker(companies, seed=seed, **settings).fit(practice.train)
        per = score_model(model, practice, board, k, seed)
        results.append((settings, float(per["ndcg_overall"].mean())))
    best = max(results, key=lambda r: r[1])[0]
    return best, results


class _RememberScores:
    """Scores each investor once, then reuses them. Safe within one dataset and board."""

    def __init__(self, model):
        self.model, self.name, self._seen = model, model.name, {}

    def fit(self, train):
        return self

    def score(self, investor_index, investor, candidates):
        if investor not in self._seen:
            self._seen[investor] = self.model.score(investor_index, investor, candidates)
        return self._seen[investor]


def select_blend_weight(train: pd.DataFrame, companies: pd.DataFrame, must_end_before: str,
                        mf_settings: dict, weights=BLEND_WEIGHTS, k: int = 10, seed: int = 0):
    """Weight on matrix factorisation (1 - weight on label similarity), chosen on the practice split."""
    practice, board = practice_split(train, must_end_before)
    mf = _RememberScores(MatrixFactorisationRanker(companies, seed=seed, **mf_settings).fit(practice.train))
    labels = _RememberScores(LabelSimilarityRanker(companies).fit(practice.train))
    results = []
    for w in weights:
        per = score_model(HybridRanker(mf, labels, w), practice, board, k, seed)
        results.append((w, float(per["ndcg_overall"].mean())))
    best = max(results, key=lambda r: r[1])[0]
    return best, results


def select_exploration(train: pd.DataFrame, companies: pd.DataFrame, must_end_before: str,
                       mf_settings: dict, blend_weight: float, strengths=EXPLORATION_STRENGTHS,
                       k: int = 10, seed: int = 0):
    """Wobble strength for Thompson sampling on the hybrid, chosen on the practice split.

    Rule (written before the first run): among strengths where new startups get at
    least their share of the practice board, pick the best practice NDCG@k. If no
    strength is fair, pick the fairest one and report that the rule could not be met.
    """
    practice, board = practice_split(train, must_end_before)
    known = set(practice.train["company_permalink"])
    hybrid = _RememberScores(HybridRanker(
        MatrixFactorisationRanker(companies, seed=seed, **mf_settings),
        LabelSimilarityRanker(companies), blend_weight).fit(practice.train))
    results = []
    for s in strengths:
        model = ThompsonRanker(hybrid, s, seed).fit(practice.train)
        per = score_model(model, practice, board, k, seed)
        ex = exposure(per, board, known)
        results.append({"strength": s, "ndcg": float(per["ndcg_overall"].mean()),
                        "new_share": ex["new_share_of_slots"], "fair": ex["new_share_of_slots"] >= ex["new_share_of_board"],
                        "board_new_share": ex["new_share_of_board"]})
    fair = [r for r in results if r["fair"]]
    chosen = max(fair, key=lambda r: r["ndcg"]) if fair else max(results, key=lambda r: r["new_share"])
    return chosen["strength"], results, bool(fair)
