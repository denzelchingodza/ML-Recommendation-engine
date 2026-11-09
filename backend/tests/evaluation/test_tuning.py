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
