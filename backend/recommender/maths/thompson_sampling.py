"""Thompson sampling, simplified: a random wobble that is bigger the less we know.

Backstory: William R. Thompson (1933) asked how to test two treatments fairly.
Do not only use the one that looks best so far, or you may never learn the
other is better. Instead, pick by luck, weighted by how sure you are.

Classic Thompson sampling keeps a full belief (a probability distribution)
about each option and updates it from live feedback. We have no live feedback,
so this is a SIMPLIFIED version:

    sampled score = score + wobble,   wobble ~ Normal(0, spread)
    spread        = strength / sqrt(1 + number of deals)

The sqrt(1 + n) shape is the usual "more evidence, less uncertainty" rule:
a brand new startup (n = 0) gets the full wobble, a startup with 3 deals gets
half of it, a startup with 99 deals a tenth.
"""

import numpy as np


def exploration_spread(n_deals, strength: float) -> np.ndarray:
    """How big the wobble is for each startup."""
    n_deals = np.asarray(n_deals, dtype=float)
    if strength < 0:
        raise ValueError(f"strength must not be negative, got {strength}")
    if (n_deals < 0).any():
        raise ValueError("number of deals cannot be negative")
    return strength / np.sqrt(1.0 + n_deals)


def sample_scores(scores, n_deals, strength: float, rng: np.random.Generator) -> np.ndarray:
    """One random draw of every startup's score."""
    scores = np.asarray(scores, dtype=float)
    spread = exploration_spread(n_deals, strength)
    if spread.shape != scores.shape:
        raise ValueError(f"scores and n_deals must have the same shape, got {scores.shape} and {spread.shape}")
    return scores + rng.normal(0.0, 1.0, size=scores.shape) * spread
