"""Pass mark 4: settings are chosen without ever seeing the hidden test."""

import pandas as pd
import pytest

from recommender.evaluation.tuning import TuningError, practice_split


def test_tuning_refuses_data_from_the_hidden_test_period():
    pairs = pd.DataFrame(
        {
            "investor_permalink": ["a", "a"],
            "company_permalink": ["x", "y"],
            "first_invested_at": pd.to_datetime(["2010-05-01", "2012-02-01"]),  # second one is test period
        }
    )
    with pytest.raises(TuningError, match="hidden test"):
        practice_split(pairs, must_end_before="2012-01-01")


# Phase 5b: each season practises on the year before its own cutoff (written BEFORE the code)


def test_practice_cutoff_can_move():
    pairs = pd.DataFrame({
        "investor_permalink": ["i"] * 7,
        "company_permalink": [f"c{n}" for n in range(7)],
        "first_invested_at": pd.to_datetime(["2007-01-01", "2007-02-01", "2007-03-01", "2007-04-01",
                                             "2007-05-01", "2008-06-01", "2008-07-01"]),
    })
    practice, board = practice_split(pairs, must_end_before="2009-01-01", practice_cutoff="2008-01-01")
    assert (practice.train.first_invested_at < pd.Timestamp("2008-01-01")).all()
    assert set(practice.test.company_permalink) == {"c5", "c6"}
    assert list(board) == ["c5", "c6"]


def test_practice_cutoff_must_be_before_the_real_cutoff():
    pairs = pd.DataFrame({"investor_permalink": ["i"], "company_permalink": ["c"],
                          "first_invested_at": pd.to_datetime(["2007-01-01"])})
    with pytest.raises(TuningError, match="practice"):
        practice_split(pairs, must_end_before="2009-01-01", practice_cutoff="2009-01-01")
