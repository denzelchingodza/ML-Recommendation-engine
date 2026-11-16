"""Phase 6 pass marks on the real data (2012 cutoff, the main scoreboard). Written BEFORE the first run."""

import pandas as pd
import pytest

from recommender.explain.report import run


@pytest.fixture(scope="module")
def output():
    return run()


def test_pass_mark_1_parts_add_up_to_the_model_score(output):
    assert output["max_add_up_error"] < 1e-9, output["max_add_up_error"]


def test_pass_mark_2_no_invented_evidence(output):
    train = output["dataset"].train
    past = train.groupby("investor_permalink")["company_permalink"].apply(set)
    labels = output["company_labels"]
    for investor, e in output["explanations"]:
        for r in e.reasons:
            assert r.examples, "a reason must cite at least one past deal"
            for company in r.examples:
                assert company in past[investor]
                assert labels[company][r.column] == r.value


def test_pass_mark_3_nothing_from_the_test_period_is_cited(output):
    test = output["dataset"].test
    hidden = set(zip(test["investor_permalink"], test["company_permalink"]))
    cutoff = pd.Timestamp(output["dataset"].split_report.cutoff)
    train = output["dataset"].train
    first_seen = train.set_index(["investor_permalink", "company_permalink"])["first_invested_at"]
    for investor, e in output["explanations"]:
        for r in e.reasons:
            for company in r.examples:
                assert (investor, company) not in hidden
                assert first_seen[(investor, company)] < cutoff


def test_pass_mark_4_every_top_10_pick_is_explained(output):
    assert output["picks"] == output["test_investors"] * output["k"]
    assert len(output["explanations"]) == output["picks"]
    for _, e in output["explanations"]:
        assert e.text
        assert e.by_chance == (e.score == 0.0)
        if e.by_chance:
            assert "by chance" in e.text


def test_pass_mark_5_biggest_reason_first(output):
    for _, e in output["explanations"]:
        parts = [r.part for r in e.reasons]
        assert parts == sorted(parts, reverse=True)
