"""Phase 5b pass marks: the multi-season scoreboard. Written BEFORE the first run.

Four seasons (cutoffs 2009, 2010, 2011, 2012). Each season learns only from deals
before its cutoff, is tested on the 12 months after it, and reruns the same tuning
recipe on the year just before its cutoff.

Pass mark 5 (whole CI run under 12 minutes) is checked on the CI log, not here.
"""

import pandas as pd
import pytest

from recommender.evaluation.seasons import FRESH_SEASONS, SEASON_CUTOFFS, combined_diff, run, run_season


@pytest.fixture(scope="module")
def output():
    return run()


def test_the_four_seasons_are_the_agreed_ones():
    assert SEASON_CUTOFFS == ("2009-01-01", "2010-01-01", "2011-01-01", "2012-01-01")
    assert FRESH_SEASONS == ("2009-01-01", "2010-01-01", "2011-01-01")


@pytest.mark.parametrize("cutoff", SEASON_CUTOFFS)
def test_pass_mark_1_no_leaks(output, cutoff):
    season = output["seasons"][cutoff]
    start, end = pd.Timestamp(cutoff), pd.Timestamp(season["end"])
    assert end == start + pd.DateOffset(years=1)
    assert (season["dataset"].train["first_invested_at"] < start).all()
    test_dates = season["dataset"].test["first_invested_at"]
    assert ((test_dates >= start) & (test_dates < end)).all()
    # tuning practised on the year just before the cutoff, never on the season itself
    assert pd.Timestamp(season["practice_cutoff"]) == start - pd.DateOffset(years=1)
    # every company on the board raised money inside the window
    board_check = season["board_dates"]
    assert ((board_check >= start) & (board_check < end)).all()


@pytest.mark.parametrize("cutoff", SEASON_CUTOFFS)
def test_pass_mark_2_enough_players(output, cutoff):
    investors = output["seasons"][cutoff]["dataset"].test["investor_permalink"].nunique()
    assert investors >= 300, f"season {cutoff} has only {investors} test investors"


def test_pass_mark_3_same_seed_same_numbers(output):
    again = run_season("2009-01-01")  # smallest season, to keep CI time down
    first = output["seasons"]["2009-01-01"]
    assert (again["settings"], again["weight"], again["strength"]) == (first["settings"], first["weight"], first["strength"])
    for model in first["results"]:
        pd.testing.assert_frame_equal(first["results"][model]["per_investor"], again["results"][model]["per_investor"])


# Recorded result of the first run (16 Nov 2026). Not tweaked after seeing it.
SEASONS_RECORDED = (
    "Phase 5b first run: thompson_hybrid minus label_similarity, ndcg@10 overall, "
    "all four seasons +0.0012 (CI -0.0029 to +0.0053, n=2346); fresh seasons 2009 to 2011 "
    "+0.0023 (CI -0.0028 to +0.0073, n=1614). Could be luck: not a proven win, so no promotion."
)


@pytest.mark.xfail(strict=True, reason=SEASONS_RECORDED)
def test_pass_mark_4a_thompson_hybrid_beats_label_similarity_over_all_seasons(output):
    diff = combined_diff(output, "label_similarity", "thompson_hybrid", "ndcg", "overall")
    assert diff.low > 0, f"combined CI {diff.low:+.4f} to {diff.high:+.4f} is not entirely above zero"


@pytest.mark.xfail(strict=True, reason=SEASONS_RECORDED)
def test_pass_mark_4b_same_verdict_on_the_fresh_seasons_only(output):
    diff = combined_diff(output, "label_similarity", "thompson_hybrid", "ndcg", "overall", cutoffs=FRESH_SEASONS)
    assert diff.low > 0, f"fresh-season CI {diff.low:+.4f} to {diff.high:+.4f} is not entirely above zero"
