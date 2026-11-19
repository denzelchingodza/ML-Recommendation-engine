"""Quality gate rules and promotion logic, on small made-up scoreboards. Written BEFORE the code.

Agreed rules (Denzel, 19 Nov 2026):
  1 proven win: combined CI over all seasons entirely above zero, and not a real loss on the fresh seasons
  2 fair to new startups: in every season, new-startup share at most 5 points below the champion's
  3 explainable: the challenger must have an exact explainer
"""

import numpy as np
import pandas as pd
import pytest

from recommender.gate.check import GateError, check_promotion
from recommender.gate.release import release_tag
from recommender.gate.rules import FAIRNESS_MAX_DROP, decide

CUTOFFS = ("2009-01-01", "2010-01-01", "2011-01-01", "2012-01-01")


def fake_output(lift: float, new_shares: dict, n: int = 400, seed: int = 0) -> dict:
    """A scoreboard where the challenger scores `lift` above the champion for every investor, plus noise."""
    rng = np.random.default_rng(seed)
    seasons = {}
    for cutoff in CUTOFFS:
        investors = [f"inv{i}" for i in range(n)]
        base = rng.random(n) * 0.1
        results = {}
        for name, shift in (("label_similarity", 0.0), ("challenger", lift)):
            per = pd.DataFrame({"investor": investors})
            for seg in ("overall", "known", "new"):
                per[f"ndcg_{seg}"] = base + shift + rng.normal(0, 0.01, n)
            results[name] = {"per_investor": per, "exposure": {"new_share_of_slots": new_shares[name]}}
        seasons[cutoff] = {"results": results}
    return {"k": 10, "seasons": seasons}


FAIR = {"label_similarity": 0.60, "challenger": 0.60}


def rule(decision, name):
    return next(r for r in decision.rules if r.name == name)


def test_a_clear_winner_passes_rules_1_and_2():
    d = decide(fake_output(0.02, FAIR), "label_similarity", "challenger", explainable={"challenger"})
    assert rule(d, "proven win").passed and rule(d, "fair to new startups").passed and rule(d, "explainable").passed
    assert d.passed


def test_no_real_difference_fails_rule_1():
    d = decide(fake_output(0.0, FAIR), "label_similarity", "challenger", explainable={"challenger"})
    assert not rule(d, "proven win").passed and not d.passed


def test_a_clear_loser_fails_rule_1():
    d = decide(fake_output(-0.02, FAIR), "label_similarity", "challenger", explainable={"challenger"})
    assert not rule(d, "proven win").passed


def test_fairness_allows_exactly_five_points_below_but_not_more():
    assert FAIRNESS_MAX_DROP == 0.05
    ok = decide(fake_output(0.02, {"label_similarity": 0.60, "challenger": 0.55}), "label_similarity", "challenger",
                explainable={"challenger"})
    bad = decide(fake_output(0.02, {"label_similarity": 0.60, "challenger": 0.549}), "label_similarity", "challenger",
                 explainable={"challenger"})
    assert rule(ok, "fair to new startups").passed
    assert not rule(bad, "fair to new startups").passed and not bad.passed


def test_no_explainer_fails_rule_3_even_with_a_big_win():
    d = decide(fake_output(0.05, FAIR), "label_similarity", "challenger", explainable=set())
    assert rule(d, "proven win").passed and not rule(d, "explainable").passed and not d.passed


def test_every_rule_is_always_reported_with_a_reason():
    d = decide(fake_output(0.0, FAIR), "label_similarity", "challenger", explainable=set())
    assert [r.name for r in d.rules] == ["proven win", "fair to new startups", "explainable"]
    assert all(r.detail for r in d.rules)


# Promotion logic: what champion.json is allowed to say


def never_run():
    raise AssertionError("the scoreboard must not run when no promotion is asked")


V1 = {"champion": "label_similarity", "version": 1}


def test_pass_mark_3_no_change_passes_without_running_the_scoreboard():
    result = check_promotion(V1, dict(V1), runner=never_run)
    assert result.passed and "no promotion" in result.report.lower()


def test_first_champion_is_recorded_without_a_contest():
    result = check_promotion(None, V1, runner=never_run)
    assert result.passed and "first" in result.report.lower()


def test_first_champion_must_be_version_1_and_explainable():
    with pytest.raises(GateError, match="version 1"):
        check_promotion(None, {"champion": "label_similarity", "version": 2}, runner=never_run)
    with pytest.raises(GateError, match="explain"):
        check_promotion(None, {"champion": "hybrid", "version": 1}, runner=never_run)


def test_pass_mark_4_unknown_model_fails_loudly():
    with pytest.raises(GateError, match="unknown model"):
        check_promotion(V1, {"champion": "magic_model", "version": 2}, runner=never_run)


def test_bad_file_fails_loudly():
    with pytest.raises(GateError, match="champion"):
        check_promotion(V1, {"version": 2}, runner=never_run)
    with pytest.raises(GateError, match="version"):
        check_promotion(V1, {"champion": "hybrid", "version": "2"}, runner=never_run)


def test_a_promotion_must_bump_the_version_by_exactly_one():
    with pytest.raises(GateError, match="version 2"):
        check_promotion(V1, {"champion": "hybrid", "version": 3}, runner=never_run)


def test_changing_the_version_without_a_new_champion_is_refused():
    with pytest.raises(GateError, match="same champion"):
        check_promotion(V1, {"champion": "label_similarity", "version": 2}, runner=never_run)


# Pass mark 5: the release only tags when the champion changes


def test_pass_mark_5_release_tag_only_on_change():
    assert release_tag(V1, dict(V1)) is None
    assert release_tag(None, V1) == "model-v1"
    assert release_tag(V1, {"champion": "hybrid", "version": 2}) == "model-v2"
