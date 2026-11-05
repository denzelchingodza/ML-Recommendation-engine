"""Phase 2 pass marks 2 to 5 on the real data. Written BEFORE the first run."""

import pandas as pd
import pytest

from recommender.evaluation.scoreboard import compare, expected_random_recall, run


@pytest.fixture(scope="module")
def output():
    return run()


def summary(output, model, metric, segment):
    return next(r.interval for r in output["results"][model]["summary"] if r.metric == metric and r.segment == segment)


def test_pass_mark_2_random_matches_the_maths(output):
    per = output["results"]["random"]["per_investor"]
    expected = expected_random_recall(per, output["k"])
    ci = summary(output, "random", "recall", "overall")
    assert ci.low <= expected <= ci.high, f"expected {expected:.4f}, CI {ci.low:.4f} to {ci.high:.4f}"


def test_pass_mark_3_most_popular_really_beats_random(output):
    diff = compare(
        output["results"]["random"]["per_investor"],
        output["results"]["most_popular"]["per_investor"],
        "ndcg",
        "overall",
    )
    assert diff.low > 0, f"difference CI {diff.low:.4f} to {diff.high:.4f} is not entirely above zero"


def test_pass_mark_4_known_and_new_are_reported_separately(output):
    per = output["results"]["most_popular"]["per_investor"]
    for segment in ("overall", "known", "new"):
        assert per[f"ndcg_{segment}"].notna().any()
    # every hidden deal is either known or new, never both, never lost
    assert (per.n_hidden_known + per.n_hidden_new == per.n_hidden_overall).all()
    assert per.n_hidden_overall.sum() == len(output["dataset"].test)


def test_pass_mark_5_same_seed_same_numbers(output):
    again = run()
    for model in ("random", "most_popular"):
        pd.testing.assert_frame_equal(
            output["results"][model]["per_investor"], again["results"][model]["per_investor"]
        )
