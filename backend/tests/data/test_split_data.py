"""Tests for split_data on a tiny hand-made table."""

import pandas as pd
import pytest

from recommender.data.split_data import SplitError, temporal_split


def make_pairs(rows):
    frame = pd.DataFrame(rows, columns=["investor_permalink", "company_permalink", "first_invested_at"])
    frame["first_invested_at"] = pd.to_datetime(frame["first_invested_at"])
    return frame


@pytest.fixture
def pairs():
    return make_pairs(
        [
            # inv_a: 3 companies before the cutoff, 2 after
            ["inv_a", "c1", "2010-01-01"],
            ["inv_a", "c2", "2010-02-01"],
            ["inv_a", "c3", "2011-12-31"],  # last day before cutoff: training
            ["inv_a", "c4", "2012-01-01"],  # cutoff day itself: test
            ["inv_a", "c9", "2012-06-01"],  # c9 never seen in training: brand new
            # inv_b: only 1 company before the cutoff, too little history
            ["inv_b", "c1", "2011-01-01"],
            ["inv_b", "c2", "2012-03-01"],
        ]
    )


def test_each_pair_lands_on_the_right_side_of_the_date(pairs):
    train, test, _ = temporal_split(pairs, "2012-01-01", min_history=3)
    assert (train.first_invested_at < pd.Timestamp("2012-01-01")).all()
    assert (test.first_invested_at >= pd.Timestamp("2012-01-01")).all()


def test_train_and_test_never_overlap(pairs):
    train, test, _ = temporal_split(pairs, "2012-01-01", min_history=3)
    train_keys = set(zip(train.investor_permalink, train.company_permalink))
    test_keys = set(zip(test.investor_permalink, test.company_permalink))
    assert train_keys.isdisjoint(test_keys)


def test_investors_with_too_little_history_leave_the_test(pairs):
    train, test, report = temporal_split(pairs, "2012-01-01", min_history=3)
    assert set(test.investor_permalink) == {"inv_a"}
    assert report.test_investors_dropped_low_history == 1
    assert report.test_pairs_before_filter == 3
    assert report.test_pairs == 2
    assert len(train) == 4  # training keeps everyone, only the TEST is filtered


def test_counts_brand_new_companies(pairs):
    _, _, report = temporal_split(pairs, "2012-01-01", min_history=3)
    # c4 and c9 never appear in training, so both test pairs are brand new
    assert report.test_pairs_new_company == 2
    assert report.new_company_share == 1.0


def test_cutoff_outside_the_data_is_refused(pairs):
    with pytest.raises(SplitError, match="inside the data range"):
        temporal_split(pairs, "1999-01-01", min_history=1)


def test_missing_columns_are_named(pairs):
    with pytest.raises(SplitError, match="missing columns"):
        temporal_split(pairs.drop(columns=["first_invested_at"]), "2012-01-01")


def test_empty_test_set_fails_loudly_instead_of_scoring_nothing(pairs):
    with pytest.raises(SplitError, match="No test pairs left"):
        temporal_split(pairs, "2012-01-01", min_history=50)
