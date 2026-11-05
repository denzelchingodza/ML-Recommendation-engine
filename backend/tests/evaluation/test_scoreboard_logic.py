"""Referee rules on tiny hand-made data: ties, seen companies, missing board companies."""

import numpy as np
import pandas as pd
import pytest

from recommender.evaluation.draft_board import DraftBoardError, build_draft_board, check_board_covers_test
from recommender.evaluation.scoreboard import rank_candidates


def test_ties_are_broken_randomly_not_alphabetically():
    candidates = np.array(["a", "b", "c", "d", "e"])
    tied = np.zeros(5)
    firsts = {rank_candidates(candidates, tied, np.random.default_rng(i))[0] for i in range(50)}
    assert len(firsts) > 1  # alphabetical would always put "a" first


def test_higher_scores_come_first():
    ranked = rank_candidates(np.array(["low", "high", "mid"]), np.array([1.0, 9.0, 5.0]), np.random.default_rng(0))
    assert list(ranked) == ["high", "mid", "low"]


def test_board_only_has_companies_raising_after_the_cutoff():
    investments = pd.DataFrame(
        {
            "company_permalink": ["old", "both", "both", "late", None],
            "funded_at": ["2010-01-01", "2010-01-01", "2012-05-01", "2013-01-01", "2013-01-01"],
        }
    )
    assert list(build_draft_board(investments, "2012-01-01")) == ["both", "late"]


def test_hidden_company_missing_from_board_fails_loudly():
    test = pd.DataFrame({"company_permalink": ["on_board", "missing"]})
    with pytest.raises(DraftBoardError, match="1 hidden test companies"):
        check_board_covers_test(np.array(["on_board"]), test)
