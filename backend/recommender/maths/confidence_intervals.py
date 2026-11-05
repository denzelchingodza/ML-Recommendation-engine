"""Bootstrap confidence intervals: is a score (or a win) real, or just luck?

The idea, with basketball:
    We cannot replay the season, so we fake many seasons from the games we have.
    1. Pick values at random from the real ones, WITH replacement, same count.
    2. Take the average of that fake season.
    3. Repeat many times.
    4. The middle 95% of those averages is the 95% confidence interval.

Checked against scipy.stats.bootstrap in tests/maths/test_confidence_intervals.py.
"""

from dataclasses import dataclass

import numpy as np

_CHUNK = 1_000  # resamples per batch, so memory stays small for big inputs


@dataclass(frozen=True)
class ConfidenceInterval:
    mean: float
    low: float
    high: float
    level: float
    n: int

    @property
    def excludes_zero(self) -> bool:
        """True when the whole interval sits on one side of zero."""
        return self.low > 0 or self.high < 0


def _check_values(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"values must be a flat list, got shape {values.shape}")
    if len(values) == 0:
        raise ValueError("values is empty: cannot build a confidence interval from nothing")
    if np.isnan(values).any():
        raise ValueError("values contains NaN: drop or explain missing scores before calling this")
    return values


def bootstrap_mean_ci(values, n_resamples: int = 10_000, level: float = 0.95, seed: int = 0) -> ConfidenceInterval:
    """Percentile bootstrap confidence interval for the mean of `values`."""
    values = _check_values(values)
    if not 0 < level < 1:
        raise ValueError(f"level must be between 0 and 1, got {level}")
    if n_resamples < 100:
        raise ValueError(f"n_resamples must be at least 100, got {n_resamples}")

    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    for start in range(0, n_resamples, _CHUNK):
        size = min(_CHUNK, n_resamples - start)
        picks = rng.integers(0, len(values), size=(size, len(values)))  # with replacement
        means[start : start + size] = values[picks].mean(axis=1)        # one average per fake season

    tail = (1 - level) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    return ConfidenceInterval(float(values.mean()), float(low), float(high), level, len(values))


def paired_bootstrap_diff_ci(values_a, values_b, n_resamples: int = 10_000, level: float = 0.95, seed: int = 0) -> ConfidenceInterval:
    """Confidence interval for mean(b - a), where a[i] and b[i] are the SAME investor.

    Pairing matters: some investors are easy for every model and some are hard.
    Comparing each investor with themselves removes that noise, so real wins
    show up more clearly. If the interval is entirely above zero, b really
    beats a.
    """
    a, b = _check_values(values_a), _check_values(values_b)
    if len(a) != len(b):
        raise ValueError(f"paired values must have the same length, got {len(a)} and {len(b)}")
    return bootstrap_mean_ci(b - a, n_resamples=n_resamples, level=level, seed=seed)
