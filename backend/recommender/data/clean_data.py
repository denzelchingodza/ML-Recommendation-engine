"""Clean the investment rows and turn them into investor to company pairs.

Every row that gets dropped is counted, so the report always says what was
removed and why. Nothing disappears silently.
"""

from dataclasses import dataclass

import pandas as pd

PAIR_COLUMNS = ["investor_permalink", "company_permalink", "first_invested_at"]


@dataclass(frozen=True)
class CleanReport:
    rows_in: int
    exact_duplicates_dropped: int
    missing_ids_dropped: int
    undated_dropped: int
    rows_kept: int
    unique_pairs: int


def clean_investments(investments: pd.DataFrame) -> tuple[pd.DataFrame, CleanReport]:
    """Return one row per (investor, company) pair with the first date they invested.

    Steps, in order:
    1. Drop rows that are exact copies of another row.
    2. Drop rows with no investor or no company id (we cannot use them).
    3. Drop rows whose date is missing or not a real date.
    4. Collapse repeat investments into one pair, keeping the FIRST date.
       Keeping the first date matters for the split: if an investor first
       backed a company before the cutoff, the pair belongs to training,
       even if they invested again later.
    """
    rows_in = len(investments)

    deduped = investments.drop_duplicates()
    exact_duplicates = rows_in - len(deduped)

    has_ids = deduped["investor_permalink"].notna() & deduped["company_permalink"].notna()
    with_ids = deduped[has_ids]
    missing_ids = len(deduped) - len(with_ids)

    dates = pd.to_datetime(with_ids["funded_at"], format="%Y-%m-%d", errors="coerce")
    dated = with_ids.assign(first_invested_at=dates).dropna(subset=["first_invested_at"])
    undated = len(with_ids) - len(dated)

    pairs = (
        dated.groupby(["investor_permalink", "company_permalink"], as_index=False)["first_invested_at"]
        .min()
        .sort_values(PAIR_COLUMNS)
        .reset_index(drop=True)
    )

    report = CleanReport(
        rows_in=rows_in,
        exact_duplicates_dropped=exact_duplicates,
        missing_ids_dropped=missing_ids,
        undated_dropped=undated,
        rows_kept=len(dated),
        unique_pairs=len(pairs),
    )
    return pairs[PAIR_COLUMNS], report
