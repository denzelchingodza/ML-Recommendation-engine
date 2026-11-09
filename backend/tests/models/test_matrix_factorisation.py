"""Matrix factorisation: pass marks 1 (gradient check) and 3 (toy test), plus safety checks."""

import numpy as np
import pandas as pd
import pytest

from recommender.models.matrix_factorisation import MatrixFactorisationRanker

COMPANIES = pd.DataFrame(
    {
        "permalink": ["sw1", "sw2", "sw3", "bio1", "bio2", "bio3", "sw_new", "bio_new", "no_labels"],
        "category_code": ["software"] * 3 + ["biotech"] * 3 + ["software", "biotech", None],
        "state_code": ["CA", "CA", "NY", "MA", "MA", "NY", "CA", "MA", None],
    }
)


def tiny_model():
    train = pd.DataFrame({"investor_permalink": ["i1", "i1", "i2", "i3"], "company_permalink": ["sw1", "sw2", "bio1", "bio2"]})
    model = MatrixFactorisationRanker(COMPANIES, dim=3, reg=0.1, epochs=1).fit(train)
    rng = np.random.default_rng(0)
    for name in ("U", "L", "C"):
        setattr(model, name, rng.normal(0, 0.5, getattr(model, name).shape))
    return model


def test_pass_mark_1_hand_gradients_match_numerical_gradients():
    model = tiny_model()
    u, p, n = np.array([0, 0, 1, 2]), np.array([0, 1, 2, 3]), np.array([2, 3, 0, 1])
    _, grads = model._batch_loss_and_grads(u, p, n)
    eps = 1e-6
    for name in ("U", "L", "C"):
        params = getattr(model, name)
        for idx in np.ndindex(params.shape):
            old = params[idx]
            params[idx] = old + eps
            up, _ = model._batch_loss_and_grads(u, p, n)
            params[idx] = old - eps
            down, _ = model._batch_loss_and_grads(u, p, n)
            params[idx] = old
            numerical = (up - down) / (2 * eps)
            assert grads[name][idx] == pytest.approx(numerical, rel=1e-5, abs=1e-8), f"{name}{idx}"


def test_pass_mark_3_toy_pattern_is_learned_including_brand_new_startups():
    # Software investors and biotech investors. After training, a NEW software
    # startup (never seen, labels only) must beat a new biotech one for the
    # software investors, and the other way round.
    rows = []
    for i in range(6):
        rows += [(f"soft{i}", c) for c in ("sw1", "sw2", "sw3")]
        rows += [(f"bio{i}", c) for c in ("bio1", "bio2", "bio3")]
    train = pd.DataFrame(rows, columns=["investor_permalink", "company_permalink"])
    model = MatrixFactorisationRanker(COMPANIES, dim=4, reg=0.01, learning_rate=0.1, epochs=200, batch_size=8).fit(train)
    candidates = np.array(["sw_new", "bio_new"])
    assert model.loss_history[-1] < model.loss_history[0]
    for i in range(6):
        soft = model.score(i, f"soft{i}", candidates)
        bio = model.score(i, f"bio{i}", candidates)
        assert soft[0] > soft[1] and bio[1] > bio[0]


def test_startups_without_labels_or_profile_score_zero():
    model = tiny_model()
    model.fit(pd.DataFrame({"investor_permalink": ["i1"], "company_permalink": ["sw1"]}))
    scores = model.score(0, "i1", np.array(["no_labels", "not_in_table"]))
    assert list(scores) == [0.0, 0.0]


def test_bad_use_fails_loudly():
    with pytest.raises(ValueError, match="at least 1"):
        MatrixFactorisationRanker(COMPANIES, dim=0)
    model = MatrixFactorisationRanker(COMPANIES)
    with pytest.raises(RuntimeError, match="before fit"):
        model.score(0, "i1", np.array(["sw1"]))
    with pytest.raises(ValueError, match="No training pairs"):
        model.fit(pd.DataFrame(columns=["investor_permalink", "company_permalink"]))
