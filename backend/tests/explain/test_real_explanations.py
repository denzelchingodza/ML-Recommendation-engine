"""Phase 6 pass marks on the real data (2012 cutoff, the main scoreboard). Written BEFORE the first run."""

import pandas as pd
import pytest

from recommender.explain.report import run


@pytest.fixture(scope="module", params=["label_similarity", "label_similarity_full"])
def output(request):
    """The five Phase 6 checks run for the champion AND the challenger (gate rule 3)."""
    return run(model_name=request.param)


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


def test_challenger_pass_mark_2_missing_label_picks_fall_below_10_percent(output):
    """Written BEFORE the first run. The champion sits at 36%; the fix must bring it under 10%."""
    if output["model_name"] != "label_similarity_full":
        pytest.skip("pass mark 2 is about the challenger")
    share = output["missing_label_picks"] / output["picks"]
    assert share < 0.10, f"{share:.1%} of picks are startups missing a label"
