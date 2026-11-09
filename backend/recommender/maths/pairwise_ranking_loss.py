"""Pairwise ranking loss (BPR): "the investor backed A, not B, so A should score higher".

For one example, let
    x = score(investor, A) - score(investor, B)      the gap between the two

The loss for that example is
    loss(x) = -log(sigmoid(x)) = log(1 + e^(-x))

    sigmoid(x) = 1 / (1 + e^(-x))  turns any number into a value between 0 and 1.
    Read it as "how sure the model is that A beats B".

    x large and positive  -> model is sure A beats B -> loss near 0 (good)
    x = 0                 -> a coin flip            -> loss = log 2, about 0.69
    x negative            -> model has it backwards -> loss grows (bad)

The slope (derivative) tells training which way to nudge:
    d loss / dx = -sigmoid(-x)

    It is always negative, so increasing the gap always lowers the loss, and the
    push is strongest when the model is most wrong.

Backstory: Steffen Rendle and colleagues, "BPR: Bayesian Personalized Ranking
from Implicit Feedback" (2009). Built for exactly our situation: we only see
who invested (positives), never an explicit "no".

Checked against scipy and a numerical slope in tests/maths/test_pairwise_ranking_loss.py.
"""

import numpy as np


def sigmoid(x):
    """1 / (1 + e^(-x)), computed without overflow for very large or small x."""
    x = np.asarray(x, dtype=float)
    return np.exp(-np.logaddexp(0.0, -x))


def bpr_loss(x):
    """log(1 + e^(-x)) for each gap x. logaddexp keeps it stable for big gaps."""
    return np.logaddexp(0.0, -np.asarray(x, dtype=float))


def bpr_loss_slope(x):
    """d loss / dx = -sigmoid(-x)."""
    return -sigmoid(-np.asarray(x, dtype=float))
