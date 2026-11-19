"""Cosine similarity: how much do two checklists point the same way?

    cosine(a, b) = (a . b) / (|a| * |b|)

    a . b   = dot product: multiply box by box, then add up. Counts shared ticks.
    |a|     = length of a: square root of (a . a).

The answer is between 0 and 1 for our checklists (no negative numbers):
1 = same direction (same mix of labels), 0 = nothing in common.

Dividing by the lengths means a big investor with 40 deals and a small one
with 5 deals can still match the same startup equally well. Only the MIX
matters, not the size.

Checked against scikit-learn in tests/maths/test_cosine_similarity.py.
"""

import numpy as np


def _check(array, name: str) -> np.ndarray:
    array = np.asarray(array, dtype=float)
    if np.isnan(array).any():
        raise ValueError(f"{name} contains NaN")
    return array


def cosine_similarity_to_rows(query, rows) -> np.ndarray:
    """Cosine similarity between one vector and every row of a matrix.

    A zero vector (no labels at all) has no direction, so cosine is undefined.
    We return 0.0 for it ("nothing in common") instead of dividing by zero.
    """
    query = _check(query, "query")
    rows = _check(rows, "rows")
    if query.ndim != 1:
        raise ValueError(f"query must be one vector, got shape {query.shape}")
    if rows.ndim != 2 or rows.shape[1] != query.shape[0]:
        raise ValueError(f"rows must have shape (n, {query.shape[0]}), got {rows.shape}")

    dots = rows @ query                                   # a . b for every row at once
    lengths = np.linalg.norm(rows, axis=1) * np.linalg.norm(query)
    out = np.zeros(len(rows))
    ok = lengths > 0
    out[ok] = dots[ok] / lengths[ok]
    return out


def cosine_similarity(a, b) -> float:
    """Cosine similarity between two vectors."""
    a = _check(a, "a")
    return float(cosine_similarity_to_rows(a, np.atleast_2d(b))[0])


def similarity_with_fixed_row_length(query, rows, row_length: float) -> np.ndarray:
    """Like cosine similarity, but every row is treated as having the same length.

        score = (a . b) / (|a| x row_length)

    Used by label_similarity_full: every startup is scored as if it had all its
    labels, so a MISSING label counts as a non-match instead of quietly making
    the checklist shorter (a shorter checklist means a smaller divisor, so one
    match would score higher than it should).

    For a row that really has row_length (both labels ticked), this is exactly cosine.
    """
    query = _check(query, "query")
    rows = _check(rows, "rows")
    if query.ndim != 1:
        raise ValueError(f"query must be one vector, got shape {query.shape}")
    if rows.ndim != 2 or rows.shape[1] != query.shape[0]:
        raise ValueError(f"rows must have shape (n, {query.shape[0]}), got {rows.shape}")
    if row_length <= 0:
        raise ValueError(f"row_length must be positive, got {row_length}")
    lengths = np.full(len(rows), np.linalg.norm(query) * row_length)
    out = np.zeros(len(rows))
    ok = lengths > 0
    out[ok] = (rows @ query)[ok] / lengths[ok]
    return out
