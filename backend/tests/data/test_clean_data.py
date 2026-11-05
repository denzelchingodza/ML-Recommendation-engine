"""Tests for clean_data on a tiny hand-made table, so we know exactly what should happen."""

import pandas as pd

from recommender.data.clean_data import clean_investments


def make_rows(rows):
    return pd.DataFrame(rows, columns=["investor_permalink", "company_permalink", "funded_at"])


def test_counts_every_kind_of_dropped_row():
    rows = make_rows(
        [
            ["inv1", "co1", "2010-01-01"],
            ["inv1", "co1", "2010-01-01"],  # exact duplicate
            [None, "co2", "2010-01-01"],  # no investor id
            ["inv2", None, "2010-01-01"],  # no company id
            ["inv2", "co3", None],  # no date
            ["inv2", "co4", "not a date"],  # bad date
            ["inv2", "co5", "2011-05-05"],
        ]
    )
    pairs, report = clean_investments(rows)
    assert report.rows_in == 7
    assert report.exact_duplicates_dropped == 1
    assert report.missing_ids_dropped == 2
    assert report.undated_dropped == 2
    assert report.rows_kept == 2
    assert report.unique_pairs == 2
    assert set(zip(pairs.investor_permalink, pairs.company_permalink)) == {("inv1", "co1"), ("inv2", "co5")}


def test_repeat_investments_keep_the_first_date():
    rows = make_rows(
        [
            ["inv1", "co1", "2013-03-01"],
            ["inv1", "co1", "2010-06-01"],  # earlier round, same pair
        ]
    )
    pairs, report = clean_investments(rows)
    assert report.unique_pairs == 1
    assert pairs.loc[0, "first_invested_at"] == pd.Timestamp("2010-06-01")


def test_output_is_sorted_and_repeatable():
    rows = make_rows([["b", "x", "2010-01-01"], ["a", "y", "2010-01-01"]])
    first, _ = clean_investments(rows)
    second, _ = clean_investments(rows.iloc[::-1])  # same rows, other order
    pd.testing.assert_frame_equal(first, second)
