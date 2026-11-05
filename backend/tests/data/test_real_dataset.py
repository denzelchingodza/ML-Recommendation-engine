"""Checks the real frozen data against the Phase 1 pass marks.

These numbers were written down BEFORE the code was run. The data is
fingerprinted, so if any number changes, something in the code changed.
"""

import pandas as pd
import pytest

from recommender.data.build_dataset import build_dataset


@pytest.fixture(scope="module")
def dataset():
    return build_dataset()


def test_pass_mark_1_loading(dataset):
    assert dataset.investment_rows == 52_870
    assert dataset.company_rows == 17_727


def test_pass_mark_2_cleaning(dataset):
    c = dataset.clean_report
    assert c.exact_duplicates_dropped == 139
    assert c.missing_ids_dropped == 2
    assert c.undated_dropped == 1
    assert c.unique_pairs == 40_977


def test_pass_mark_3_split(dataset):
    train, test, s = dataset.train, dataset.test, dataset.split_report
    cutoff = pd.Timestamp("2012-01-01")
    assert (train.first_invested_at < cutoff).all()
    assert (test.first_invested_at >= cutoff).all()
    train_keys = set(zip(train.investor_permalink, train.company_permalink))
    test_keys = set(zip(test.investor_permalink, test.company_permalink))
    assert train_keys.isdisjoint(test_keys)
    history = train.groupby("investor_permalink").size()
    assert (history.loc[test.investor_permalink.unique()] >= 5).all()
    assert s.train_pairs == 28_614
    assert s.test_pairs == 5_878
    assert s.test_investors == 828


def test_pass_mark_4_repeatable(dataset):
    again = build_dataset()
    pd.testing.assert_frame_equal(dataset.train, again.train)
    pd.testing.assert_frame_equal(dataset.test, again.test)
