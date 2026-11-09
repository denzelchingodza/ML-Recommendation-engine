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
    for model in ("random", "most_popular", "label_similarity", "matrix_factorisation", "hybrid"):
        pd.testing.assert_frame_equal(
            output["results"][model]["per_investor"], again["results"][model]["per_investor"]
        )


# Phase 3 pass marks (written before the first run)


@pytest.mark.parametrize("segment", ["new", "overall"])
def test_phase3_label_similarity_really_beats_random(output, segment):
    diff = compare(
        output["results"]["random"]["per_investor"],
        output["results"]["label_similarity"]["per_investor"],
        "ndcg",
        segment,
    )
    assert diff.low > 0, f"{segment}: difference CI {diff.low:.4f} to {diff.high:.4f} is not entirely above zero"


def test_phase3_shuffled_labels_collapse_to_random(output):
    """Leak check: with labels scrambled between companies, the real signal is gone.

    If the score stayed high, the model would be learning from something other
    than the labels (a leak), and every Phase 3 result would be suspect.
    """
    import numpy as np

    from recommender.data.load_data import load_investments
    from recommender.evaluation.draft_board import build_draft_board
    from recommender.evaluation.scoreboard import score_model, summarise
    from recommender.models.label_similarity import LabelSimilarityRanker

    dataset = output["dataset"]
    shuffled = dataset.companies.copy()
    rng = np.random.default_rng(99)
    for column in ("category_code", "state_code"):
        shuffled[column] = rng.permutation(shuffled[column].to_numpy())
    board = build_draft_board(load_investments(), dataset.split_report.cutoff)
    per = score_model(LabelSimilarityRanker(shuffled).fit(dataset.train), dataset, board)

    shuffled_ci = next(r.interval for r in summarise(per) if r.metric == "ndcg" and r.segment == "overall")
    random_ci = summary(output, "random", "ndcg", "overall")
    real_ci = summary(output, "label_similarity", "ndcg", "overall")
    assert shuffled_ci.low <= random_ci.high, "shuffled labels still beat random: something is leaking"
    assert shuffled_ci.high < real_ci.low, "real labels do not clearly beat shuffled labels"


# Phase 4 pass marks (written before the first run)


def test_phase4_pass_mark_2_training_loss_goes_down(output):
    history = output["mf_model"].loss_history
    assert history[-1] < history[0] * 0.7, f"loss only went from {history[0]:.4f} to {history[-1]:.4f}"


def test_phase4_pass_mark_4_settings_come_from_the_practice_grid(output):
    from recommender.evaluation.tuning import GRID

    assert output["mf_settings"] in GRID
    assert len(output["mf_tuning"]) == len(GRID)


def test_phase4_pass_mark_5_scrambled_history_collapses_to_random(output):
    import numpy as np

    from recommender.data.load_data import load_investments
    from recommender.evaluation.draft_board import build_draft_board
    from recommender.evaluation.scoreboard import score_model, summarise
    from recommender.models.matrix_factorisation import MatrixFactorisationRanker

    dataset = output["dataset"]
    scrambled = dataset.train.copy()
    scrambled["company_permalink"] = np.random.default_rng(5).permutation(scrambled["company_permalink"].to_numpy())
    model = MatrixFactorisationRanker(dataset.companies, **output["mf_settings"]).fit(scrambled)
    board = build_draft_board(load_investments(), dataset.split_report.cutoff)
    per = score_model(model, dataset, board)
    scrambled_ci = next(r.interval for r in summarise(per) if r.metric == "ndcg" and r.segment == "overall")
    assert scrambled_ci.low <= summary(output, "random", "ndcg", "overall").high, "scrambled history still beats random"


RECORDED_LOSS = (
    "Recorded result, 9 Nov 2026: matrix factorisation ALONE did not beat label similarity "
    "(overall a tie, slightly worse on new startups). It did not become the starter. "
    "Kept so the loss stays visible; strict=True makes this fail loudly if it ever starts passing."
)


@pytest.mark.xfail(strict=True, reason=RECORDED_LOSS)
def test_phase4_pass_mark_7_beats_label_similarity_overall(output):
    diff = compare(
        output["results"]["label_similarity"]["per_investor"],
        output["results"]["matrix_factorisation"]["per_investor"],
        "ndcg",
        "overall",
    )
    assert diff.low > 0, f"difference CI {diff.low:.4f} to {diff.high:.4f} is not entirely above zero"


@pytest.mark.xfail(strict=True, reason=RECORDED_LOSS)
def test_phase4_pass_mark_8_not_clearly_worse_on_new_startups(output):
    diff = compare(
        output["results"]["label_similarity"]["per_investor"],
        output["results"]["matrix_factorisation"]["per_investor"],
        "ndcg",
        "new",
    )
    assert diff.high >= 0, f"clearly worse on new startups: CI {diff.low:.4f} to {diff.high:.4f}"


# Phase 4 hybrid pass marks (written before the first hybrid run, AFTER seeing matrix factorisation's results)


def test_hybrid_pass_mark_1_weight_chosen_on_practice_grid(output):
    from recommender.evaluation.tuning import BLEND_WEIGHTS

    assert output["blend_weight"] in BLEND_WEIGHTS
    assert len(output["blend_tuning"]) == len(BLEND_WEIGHTS)


def _hybrid_vs(output, base, segment):
    return compare(output["results"][base]["per_investor"], output["results"]["hybrid"]["per_investor"], "ndcg", segment)


HYBRID_RECORDED_LOSS = (
    "Recorded result, 9 Nov 2026: the hybrid scored 0.0203 vs 0.0171 for label similarity, but the "
    "paired CI (-0.0016 to +0.0080) crosses zero, so the win is not proven. It did not become the starter."
)


@pytest.mark.xfail(strict=True, reason=HYBRID_RECORDED_LOSS)
def test_hybrid_pass_mark_2_beats_label_similarity_overall(output):
    diff = _hybrid_vs(output, "label_similarity", "overall")
    assert diff.low > 0, f"CI {diff.low:.4f} to {diff.high:.4f}"


def test_hybrid_pass_mark_3_not_clearly_worse_on_new_than_label_similarity(output):
    diff = _hybrid_vs(output, "label_similarity", "new")
    assert diff.high >= 0, f"CI {diff.low:.4f} to {diff.high:.4f}"


def test_hybrid_pass_mark_4_not_clearly_worse_on_known_than_matrix_factorisation(output):
    diff = _hybrid_vs(output, "matrix_factorisation", "known")
    assert diff.high >= 0, f"CI {diff.low:.4f} to {diff.high:.4f}"
