"""Load the raw Crunchbase CSV files and check they are exactly what we expect.

Every check fails loudly with a clear message. Loading never quietly carries
on with the wrong data.
"""

import hashlib
from pathlib import Path

import pandas as pd

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# The files are Mac Roman, not UTF-8. See data/raw/SOURCE.md.
ENCODING = "mac_roman"

INVESTMENTS_FILE = "crunchbase-investments.csv"
COMPANIES_FILE = "crunchbase-companies.csv"

EXPECTED_FINGERPRINTS = {
    INVESTMENTS_FILE: "8f13f28d35915633c0a814dd22d49540ee115717d10d387ebc3a49183a35b65f",
    COMPANIES_FILE: "83217e8165db15124aec2799f5f28bda910ab09cc1ed9553b066eb97f60f195d",
}

INVESTMENT_COLUMNS = [
    "company_permalink",
    "company_name",
    "company_category_code",
    "company_city",
    "investor_permalink",
    "investor_name",
    "funding_round_type",
    "funded_at",
]

COMPANY_COLUMNS = [
    "permalink",
    "name",
    "category_code",
    "city",
    "country_code",
]


class DataLoadError(Exception):
    """Raised when a data file is missing, changed or has the wrong shape."""


def file_fingerprint(path: Path) -> str:
    """Return the SHA-256 fingerprint of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv(path: Path, expected_fingerprint: str, required_columns: list[str]) -> pd.DataFrame:
    """Load one CSV after checking it exists, is unchanged and has the needed columns."""
    path = Path(path)
    if not path.is_file():
        raise DataLoadError(f"Could not find data file: {path}")

    actual = file_fingerprint(path)
    if actual != expected_fingerprint:
        raise DataLoadError(
            f"{path.name} has changed. Expected fingerprint {expected_fingerprint}, "
            f"got {actual}. Refusing to load data that is not the frozen copy."
        )

    try:
        frame = pd.read_csv(path, encoding=ENCODING, dtype=str, keep_default_na=True)
    except Exception as error:  # report the real cause, never hide it
        raise DataLoadError(f"Could not read {path.name}: {error}") from error

    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise DataLoadError(f"{path.name} is missing required columns: {missing}")

    if frame.empty:
        raise DataLoadError(f"{path.name} loaded but has no rows")

    return frame


def load_investments(data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load every investment row: one row per investor per funding round."""
    return load_csv(
        Path(data_dir) / INVESTMENTS_FILE,
        EXPECTED_FINGERPRINTS[INVESTMENTS_FILE],
        INVESTMENT_COLUMNS,
    )


def load_companies(data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load every company row: one row per company."""
    return load_csv(
        Path(data_dir) / COMPANIES_FILE,
        EXPECTED_FINGERPRINTS[COMPANIES_FILE],
        COMPANY_COLUMNS,
    )
