"""Label similarity: recommend startups whose labels match an investor's past deals.

1. Every company becomes a checklist (one-hot vector): one tick for its
   category, one tick for its state. A missing label means no tick.
2. Every investor becomes a checklist too: the checklists of all the companies
   they backed BEFORE the cutoff, added up. For example 17 ticks on software,
   30 on California.
3. A startup's score = cosine similarity between its checklist and the
   investor's. Brand new startups have labels, so they get real scores.

Labels come from the company table, which describes companies (category and
state). It holds no information about who invested, so it does not leak the
test answers.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from recommender.maths.cosine_similarity import cosine_similarity_to_rows

LABEL_COLUMNS = ("category_code", "state_code")


@dataclass(frozen=True)
class CompanyVectors:
    index: dict          # company permalink -> row number
    matrix: np.ndarray   # one row per company, one column per label
    labels: list         # column names, e.g. "category_code=software"
    groups: dict         # label column -> list of its column numbers


def build_company_vectors(companies: pd.DataFrame, columns=LABEL_COLUMNS) -> CompanyVectors:
    """One-hot checklist for every company."""
    missing = [c for c in ("permalink", *columns) if c not in companies.columns]
    if missing:
        raise ValueError(f"companies is missing columns: {missing}")
    if companies["permalink"].duplicated().any():
        raise ValueError("companies has duplicate permalinks, so a company would get two checklists")

    labels, groups = [], {}
    for column in columns:
        values = sorted(companies[column].dropna().unique())
        groups[column] = list(range(len(labels), len(labels) + len(values)))
        labels += [f"{column}={v}" for v in values]
    position = {label: i for i, label in enumerate(labels)}

    matrix = np.zeros((len(companies), len(labels)), dtype=np.float32)
    for column in columns:
        values = companies[column].to_numpy()
        rows = np.flatnonzero(pd.notna(values))
        cols = [position[f"{column}={v}"] for v in values[rows]]
        matrix[rows, cols] = 1.0

    index = {permalink: i for i, permalink in enumerate(companies["permalink"])}
    return CompanyVectors(index, matrix, labels, groups)


class LabelSimilarityRanker:
    """Score startups by how well their labels match the investor's past deals."""

    name = "label_similarity"

    def __init__(self, companies: pd.DataFrame, columns=LABEL_COLUMNS):
        self.vectors = build_company_vectors(companies, columns)
        self.profiles: dict | None = None

    def _rows(self, permalinks) -> np.ndarray:
        """Checklists for these companies. A company missing from the table gets an empty checklist."""
        out = np.zeros((len(permalinks), self.vectors.matrix.shape[1]), dtype=np.float32)
        for i, p in enumerate(permalinks):
            row = self.vectors.index.get(p)
            if row is not None:
                out[i] = self.vectors.matrix[row]
        return out

    def fit(self, train: pd.DataFrame) -> "LabelSimilarityRanker":
        pairs = train[["investor_permalink", "company_permalink"]].drop_duplicates()
        self.profiles = {
            investor: self._rows(group["company_permalink"].to_numpy()).sum(axis=0)
            for investor, group in pairs.groupby("investor_permalink")
        }
        return self

    def score(self, investor_index: int, investor: str, candidates: np.ndarray) -> np.ndarray:
        if self.profiles is None:
            raise RuntimeError("LabelSimilarityRanker.score called before fit")
        profile = self.profiles.get(investor)
        if profile is None:
            raise KeyError(f"No training history for investor {investor!r}, so there is no profile to match against")
        return cosine_similarity_to_rows(profile, self._rows(candidates))
