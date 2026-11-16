"""Thompson sampling on top of any model: give less-known startups a fair shot.

Used in our project on the hybrid, which is confident about known startups and
unsure about new ones, and so buries new ones (22% of top-10 slots, while they
are 67% of the board).

The base model's scores are standardised first, so `strength` means the same
thing whatever model sits underneath. Each investor gets their own seeded
random draw, so results repeat exactly.
"""

import numpy as np
import pandas as pd

from recommender.maths.score_scaling import standardise
from recommender.maths.thompson_sampling import sample_scores


class ThompsonRanker:
    def __init__(self, base_model, strength: float, seed: int = 0):
        if strength < 0:
            raise ValueError(f"strength must not be negative, got {strength}")
        self.base_model, self.strength, self.seed = base_model, strength, seed
        self.name = f"thompson_{base_model.name}"
        self.deal_counts: pd.Series | None = None

    def fit(self, train: pd.DataFrame) -> "ThompsonRanker":
        self.base_model.fit(train)
        # Evidence per startup: how many different investors backed it in training.
        self.deal_counts = train.groupby("company_permalink")["investor_permalink"].nunique()
        return self

    def score(self, investor_index: int, investor: str, candidates: np.ndarray) -> np.ndarray:
        if self.deal_counts is None:
            raise RuntimeError("ThompsonRanker.score called before fit")
        base = standardise(self.base_model.score(investor_index, investor, candidates))
        n_deals = self.deal_counts.reindex(candidates, fill_value=0).to_numpy()
        rng = np.random.default_rng([self.seed, 7, investor_index])
        return sample_scores(base, n_deals, self.strength, rng)
