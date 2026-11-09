"""Hybrid: mix two models' scores, so each covers the other's weak spot.

    score = w * standardised(model A) + (1 - w) * standardised(model B)

In our project: A = matrix factorisation (strong on known startups),
B = label similarity (strong on new startups). The weight w is chosen on the
practice split only, never on the hidden test.
"""

import numpy as np

from recommender.maths.score_scaling import standardise


class HybridRanker:
    name = "hybrid"

    def __init__(self, model_a, model_b, weight: float):
        if not 0 <= weight <= 1:
            raise ValueError(f"weight must be between 0 and 1, got {weight}")
        self.model_a, self.model_b, self.weight = model_a, model_b, weight

    def fit(self, train):
        self.model_a.fit(train)
        self.model_b.fit(train)
        return self

    def score(self, investor_index: int, investor: str, candidates: np.ndarray) -> np.ndarray:
        a = standardise(self.model_a.score(investor_index, investor, candidates))
        b = standardise(self.model_b.score(investor_index, investor, candidates))
        return self.weight * a + (1 - self.weight) * b
