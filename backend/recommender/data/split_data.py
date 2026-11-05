"""Split investor to company pairs into training and hidden test sets by date.

One cutoff date for everyone:
- pairs first invested BEFORE the cutoff go to training
- pairs first invested ON or AFTER the cutoff go to the hidden test set

This stops the model from learning anything that happened after the cutoff.
A random split would let it peek at the future (leakage).
"""

from dataclasses import dataclass

import pandas as pd

DEFAULT_CUTOFF = "2012-01-01"
DEFAULT_MIN_HISTORY = 5


class SplitError(Exception):
    """Raised when a split cannot be made honestly."""


@dataclass(frozen=True)
class SplitReport:
    cutoff: str
    min_history: int
    train_pairs: int
    test_pairs_before_filter: int
    test_investors_dropped_low_history: int
    test_pairs: int
    test_investors: int
    test_pairs_new_company: int

    @property
    def new_company_share(self) -> float:
        """Share of hidden test pairs whose company never appears in training."""
        return self.test_pairs_new_company / self.test_pairs


def temporal_split(
    pairs: pd.DataFrame,
    cutoff: str = DEFAULT_CUTOFF,
    min_history: int = DEFAULT_MIN_HISTORY,
) -> tuple[pd.DataFrame, pd.DataFrame, SplitReport]:
    """Split pairs at a single cutoff date.

    Only investors with at least `min_history` training companies are kept in
    the test set: with less history there is nothing to learn their taste from.
    """
    required = {"investor_permalink", "company_permalink", "first_invested_at"}
    missing = required - set(pairs.columns)
    if missing:
        raise SplitError(f"pairs is missing columns: {sorted(missing)}")
    if min_history < 1:
        raise SplitError(f"min_history must be at least 1, got {min_history}")

    cutoff_date = pd.Timestamp(cutoff)
    earliest, latest = pairs["first_invested_at"].min(), pairs["first_invested_at"].max()
    if not earliest < cutoff_date <= latest:
        raise SplitError(
            f"Cutoff {cutoff} must fall inside the data range ({earliest.date()} to {latest.date()})"
        )

    is_train = pairs["first_invested_at"] < cutoff_date
    train = pairs[is_train]
    test_all = pairs[~is_train]

    history = train.groupby("investor_permalink").size()
    eligible = set(history[history >= min_history].index)
    test = test_all[test_all["investor_permalink"].isin(eligible)]

    if test.empty:
        raise SplitError(
            f"No test pairs left after keeping investors with {min_history}+ training companies"
        )

    dropped_investors = (
        test_all.loc[~test_all["investor_permalink"].isin(eligible), "investor_permalink"].nunique()
    )
    new_company = ~test["company_permalink"].isin(set(train["company_permalink"]))

    sort_cols = ["investor_permalink", "company_permalink"]
    train = train.sort_values(sort_cols).reset_index(drop=True)
    test = test.sort_values(sort_cols).reset_index(drop=True)

    report = SplitReport(
        cutoff=cutoff,
        min_history=min_history,
        train_pairs=len(train),
        test_pairs_before_filter=len(test_all),
        test_investors_dropped_low_history=dropped_investors,
        test_pairs=len(test),
        test_investors=test["investor_permalink"].nunique(),
        test_pairs_new_company=int(new_company.sum()),
    )
    return train, test, report
