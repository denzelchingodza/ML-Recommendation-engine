"""Phase 7 pass marks 1 and 2 on the real multi-season scoreboard. Written BEFORE the first run."""

import numpy as np
import pytest

from recommender.evaluation.scoreboard_core import exposure, score_model
from recommender.evaluation.seasons import run
from recommender.gate.registry import EXPLAINABLE
from recommender.gate.rules import decide


@pytest.fixture(scope="module")
def output():
    return run()


def failed(decision):
    return {r.name for r in decision.rules if not r.passed}


def test_pass_mark_1_rejects_matrix_factorisation(output):
    d = decide(output, "label_similarity", "matrix_factorisation", EXPLAINABLE)
    assert not d.passed and {"proven win", "explainable"} <= failed(d)


def test_pass_mark_1_rejects_the_hybrid(output):
    d = decide(output, "label_similarity", "hybrid", EXPLAINABLE)
    assert not d.passed and {"proven win", "fair to new startups", "explainable"} <= failed(d)


def test_pass_mark_1_rejects_thompson_on_the_hybrid(output):
    d = decide(output, "label_similarity", "thompson_hybrid", EXPLAINABLE)
    assert not d.passed and {"proven win", "explainable"} <= failed(d)


class PeeksAtTheAnswers:
    """A cheat: scores 1 for the startups the investor really backed in the test window."""

    name = "cheat"

    def __init__(self, test):
        self.hidden = test.groupby("investor_permalink")["company_permalink"].apply(set)

    def fit(self, train):
        return self

    def score(self, investor_index, investor, candidates):
        hidden = self.hidden.get(investor, set())
        return np.array([1.0 if c in hidden else 0.0 for c in candidates])


def test_pass_mark_2_the_gate_can_say_yes(output):
    seasons = {}
    for cutoff, season in output["seasons"].items():
        cheat = PeeksAtTheAnswers(season["dataset"].test)
        per = score_model(cheat, season["dataset"], season["board"], output["k"])
        known = set(season["dataset"].train["company_permalink"])
        results = dict(season["results"], cheat={"per_investor": per, "exposure": exposure(per, season["board"], known)})
        seasons[cutoff] = dict(season, results=results)
    d = decide({"k": output["k"], "seasons": seasons}, "label_similarity", "cheat", EXPLAINABLE | {"cheat"})
    assert "proven win" not in failed(d), [r.detail for r in d.rules]
