"""The two starters every real model must beat.

Each model gives every candidate company a score (higher = recommend first).
The evaluation code turns scores into a ranking and breaks ties randomly.
"""

import numpy as np
import pandas as pd


class RandomRanker:
    """The blindfolded scout: a random score for every company."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, train: pd.DataFrame) -> "RandomRanker":
        return self  # learns nothing, on purpose

    def score(self, investor_index: int, investor: str, candidates: np.ndarray) -> np.ndarray:
        # Seeded per investor, so results do not depend on the order investors are scored in.
        rng = np.random.default_rng([self.seed, investor_index])
        return rng.random(len(candidates))


class MostPopularRanker:
    """Rank companies by how many different investors backed them BEFORE the cutoff.

    Brand new companies have no history, so they all score 0 and tie at the
    bottom. That is this model's known weakness.
    """

    name = "most_popular"

    def __init__(self):
        self.popularity: pd.Series | None = None

    def fit(self, train: pd.DataFrame) -> "MostPopularRanker":
        self.popularity = train.groupby("company_permalink")["investor_permalink"].nunique()
        return self

    def score(self, investor_index: int, investor: str, candidates: np.ndarray) -> np.ndarray:
        if self.popularity is None:
            raise RuntimeError("MostPopularRanker.score called before fit")
        return self.popularity.reindex(candidates, fill_value=0).to_numpy(dtype=float)
