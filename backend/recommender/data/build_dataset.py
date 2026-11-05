"""Run Phase 1 end to end: load, clean, split, then print the report.

Run it from the backend folder:
    python -m recommender.data.build_dataset
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from recommender.data.clean_data import CleanReport, clean_investments
from recommender.data.load_data import RAW_DATA_DIR, load_companies, load_investments
from recommender.data.split_data import (
    DEFAULT_CUTOFF,
    DEFAULT_MIN_HISTORY,
    SplitReport,
    temporal_split,
)


@dataclass(frozen=True)
class Dataset:
    train: pd.DataFrame
    test: pd.DataFrame
    companies: pd.DataFrame
    investment_rows: int
    company_rows: int
    clean_report: CleanReport
    split_report: SplitReport


def build_dataset(
    data_dir: Path = RAW_DATA_DIR,
    cutoff: str = DEFAULT_CUTOFF,
    min_history: int = DEFAULT_MIN_HISTORY,
) -> Dataset:
    investments = load_investments(data_dir)
    companies = load_companies(data_dir)
    pairs, clean_report = clean_investments(investments)
    train, test, split_report = temporal_split(pairs, cutoff, min_history)
    return Dataset(
        train=train,
        test=test,
        companies=companies,
        investment_rows=len(investments),
        company_rows=len(companies),
        clean_report=clean_report,
        split_report=split_report,
    )


def format_report(dataset: Dataset) -> str:
    c, s = dataset.clean_report, dataset.split_report
    return "\n".join(
        [
            "Phase 1 data report",
            "",
            "Loaded",
            f"  investment rows:            {dataset.investment_rows:,}",
            f"  company rows:               {dataset.company_rows:,}",
            "",
            "Cleaned",
            f"  exact duplicates dropped:   {c.exact_duplicates_dropped:,}",
            f"  rows missing ids dropped:   {c.missing_ids_dropped:,}",
            f"  undated rows dropped:       {c.undated_dropped:,}",
            f"  unique investor-company pairs: {c.unique_pairs:,}",
            "",
            f"Split at {s.cutoff} (test investors need {s.min_history}+ training companies)",
            f"  training pairs:             {s.train_pairs:,}",
            f"  test pairs before filter:   {s.test_pairs_before_filter:,}",
            f"  investors left out of test: {s.test_investors_dropped_low_history:,} (too little history)",
            f"  hidden test pairs:          {s.test_pairs:,}",
            f"  test investors:             {s.test_investors:,}",
            f"  test pairs with a brand new company: {s.test_pairs_new_company:,} "
            f"({s.new_company_share:.0%})",
        ]
    )


def main() -> None:
    print(format_report(build_dataset()))


if __name__ == "__main__":
    main()
