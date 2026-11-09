"""Put two models' scores on the same scale before mixing them.

Different models score on different scales (cosine between 0 and 1, matrix
factorisation any number). Adding them raw would let the bigger scale win.

Standardise (z-score) each investor's list of scores:
    z = (score - average) / spread        spread = standard deviation

After this, every model's scores have average 0 and spread 1, so a weight w
really means "how much we trust this model".

Checked against scipy.stats.zscore in tests/maths/test_score_scaling.py.
"""

import numpy as np


def standardise(scores) -> np.ndarray:
    """z-scores. If every score is equal (no spread), there is nothing to rank: return zeros."""
    scores = np.asarray(scores, dtype=float)
    if np.isnan(scores).any():
        raise ValueError("scores contains NaN")
    spread = scores.std()
    if spread == 0:
        return np.zeros_like(scores)
    return (scores - scores.mean()) / spread
