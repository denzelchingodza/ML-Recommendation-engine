"""Matrix factorisation with label profiles, trained with the pairwise ranking loss.

Every investor gets a hidden profile: a short list of numbers (sliders) the
model learns by itself.

A startup's profile is built from parts:
    startup profile = its category's profile + its state's profile
                      + its own personal profile (only if it had deals in training)

So brand new startups still get a profile from their labels.

    score(investor, startup) = investor profile . startup profile   (dot product)

Training repeats, many times:
    1. Take a real deal: investor u backed startup A.
    2. Pick a random startup B from training as a "probably not".
    3. Gap x = score(u, A) - score(u, B).
    4. Nudge every profile involved a little so the gap grows (gradient descent).

The nudges (gradients) are written by hand below and checked against a slow
numerical version in tests/models/test_matrix_factorisation.py.
"""

import numpy as np
import pandas as pd

from recommender.maths.pairwise_ranking_loss import bpr_loss, bpr_loss_slope
from recommender.models.label_similarity import LABEL_COLUMNS, build_company_vectors


class MatrixFactorisationRanker:
    name = "matrix_factorisation"

    def __init__(self, companies: pd.DataFrame, dim: int = 16, reg: float = 0.01,
                 learning_rate: float = 0.05, epochs: int = 30, batch_size: int = 256,
                 seed: int = 0, columns=LABEL_COLUMNS):
        if dim < 1 or epochs < 1 or batch_size < 1:
            raise ValueError("dim, epochs and batch_size must all be at least 1")
        if learning_rate <= 0 or reg < 0:
            raise ValueError("learning_rate must be positive and reg must not be negative")
        self.vectors = build_company_vectors(companies, columns)
        self.dim, self.reg, self.learning_rate = dim, reg, learning_rate
        self.epochs, self.batch_size, self.seed = epochs, batch_size, seed
        self.loss_history: list[float] = []
        self._fitted = False

    # ---------- the maths of one batch ----------

    def _batch_loss_and_grads(self, u, p, n):
        """Total loss over a batch of (investor u, backed p, not-backed n) triples, and its gradients.

        Loss = sum over the batch of log(1 + e^(-x))
             + reg/2 * sum over the batch of (|U_u|^2 + |C_p|^2 + |C_n|^2)
             + reg/2 * |L|^2

        A SUM, not an average: each example then gets a full-size nudge, the
        standard way BPR is trained. (Averaging made every nudge batch_size
        times too small, and the model barely learned.)

        U = investor profiles, L = label profiles, C = personal startup profiles.
        Startup profile = (its label ticks) @ L + C.
        """
        U, L, C, ticks = self.U, self.L, self.C, self.known_ticks
        Uu = U[u]
        P = ticks[p] @ L + C[p]
        N = ticks[n] @ L + C[n]
        x = np.sum(Uu * (P - N), axis=1)  # gap for each triple

        loss = (np.sum(bpr_loss(x))
                + self.reg / 2 * np.sum(np.sum(Uu ** 2, 1) + np.sum(C[p] ** 2, 1) + np.sum(C[n] ** 2, 1))
                + self.reg / 2 * np.sum(L ** 2))

        g = bpr_loss_slope(x)               # d(loss)/dx for each triple
        dP = g[:, None] * Uu                # x = Uu.P - Uu.N, so dx/dP = Uu
        dN = -g[:, None] * Uu               # and dx/dN = -Uu
        dUu = g[:, None] * (P - N) + self.reg * Uu

        dU = np.zeros_like(U)
        np.add.at(dU, u, dUu)
        dC = np.zeros_like(C)
        np.add.at(dC, p, dP + self.reg * C[p])  # P = ticks @ L + C, so dP/dC = 1
        np.add.at(dC, n, dN + self.reg * C[n])
        dL = ticks[p].T @ dP + ticks[n].T @ dN + self.reg * L  # each ticked label gets the startup's nudge
        return float(loss), {"U": dU, "L": dL, "C": dC}

    # ---------- training ----------

    def fit(self, train: pd.DataFrame) -> "MatrixFactorisationRanker":
        pairs = train[["investor_permalink", "company_permalink"]].drop_duplicates()
        if pairs.empty:
            raise ValueError("No training pairs: nothing to learn from")
        investors = sorted(pairs["investor_permalink"].unique())
        known = sorted(pairs["company_permalink"].unique())
        self.investor_index = {v: i for i, v in enumerate(investors)}
        self.known_index = {c: j for j, c in enumerate(known)}

        rows = pd.Index(self.vectors.index.keys()).get_indexer(known)
        self.known_ticks = np.zeros((len(known), self.vectors.matrix.shape[1]))
        self.known_ticks[rows >= 0] = self.vectors.matrix[rows[rows >= 0]]

        rng = np.random.default_rng(self.seed)
        scale = 0.1 / np.sqrt(self.dim)
        self.U = rng.normal(0, scale, (len(investors), self.dim))
        self.L = rng.normal(0, scale, (len(self.vectors.labels), self.dim))
        self.C = rng.normal(0, scale, (len(known), self.dim))

        u_all = pairs["investor_permalink"].map(self.investor_index).to_numpy()
        p_all = pairs["company_permalink"].map(self.known_index).to_numpy()
        self.loss_history = []
        for _ in range(self.epochs):
            order = rng.permutation(len(u_all))
            total = 0.0
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                n = rng.integers(0, len(known), size=len(batch))  # random "probably not" startups
                loss, grads = self._batch_loss_and_grads(u_all[batch], p_all[batch], n)
                for name, grad in grads.items():
                    setattr(self, name, getattr(self, name) - self.learning_rate * grad)
                total += loss
            self.loss_history.append(total / len(order))

        # Precompute every startup's profile: labels for all, plus personal profile if known.
        self._all_profiles = self.vectors.matrix @ self.L
        table_rows = pd.Index(self.vectors.index.keys()).get_indexer(known)
        self._all_profiles[table_rows[table_rows >= 0]] += self.C[table_rows >= 0]
        self._table_index = pd.Index(self.vectors.index.keys())
        self._fitted = True
        return self

    def score(self, investor_index: int, investor: str, candidates: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("MatrixFactorisationRanker.score called before fit")
        if investor not in self.investor_index:
            raise KeyError(f"No training history for investor {investor!r}, so there is no profile to match against")
        rows = self._table_index.get_indexer(candidates)
        scores = np.zeros(len(candidates))
        ok = rows >= 0  # a startup missing from the company table has no profile: score 0
        scores[ok] = self._all_profiles[rows[ok]] @ self.U[self.investor_index[investor]]
        return scores
