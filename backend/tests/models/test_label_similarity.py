"""Label similarity model on tiny hand-made companies."""

import numpy as np
import pandas as pd
import pytest

from recommender.models.label_similarity import LabelSimilarityRanker, build_company_vectors

COMPANIES = pd.DataFrame(
    {
        "permalink": ["sw_ca", "sw_ny", "bio_ca", "bio_ny", "no_labels"],
        "category_code": ["software", "software", "biotech", "biotech", None],
        "state_code": ["CA", "NY", "CA", "NY", None],
    }
)


def test_pass_mark_2_one_tick_per_group_or_none_if_missing():
    v = build_company_vectors(COMPANIES)
    for column, cols in v.groups.items():
        ticks = v.matrix[:, cols].sum(axis=1)
        has_label = COMPANIES[column].notna().to_numpy()
        assert (ticks[has_label] == 1).all()
        assert (ticks[~has_label] == 0).all()
    assert v.matrix.shape == (5, 4)  # 2 categories + 2 states


def test_duplicate_companies_fail_loudly():
    with pytest.raises(ValueError, match="duplicate"):
        build_company_vectors(pd.concat([COMPANIES, COMPANIES.iloc[:1]]))


def test_profile_ranks_the_closest_match_first():
    train = pd.DataFrame({"investor_permalink": ["inv"] * 3, "company_permalink": ["sw_ca", "sw_ca", "sw_ny"]})
    model = LabelSimilarityRanker(COMPANIES).fit(train)
    # repeat rows count once: profile = software x2, CA x1, NY x1
    candidates = np.array(["bio_ny", "sw_ca", "no_labels", "not_in_table"])
    scores = model.score(0, "inv", candidates)
    assert scores[1] > scores[0]                 # software in CA beats biotech in NY
    assert scores[2] == 0.0 and scores[3] == 0.0  # no labels -> 0, never a crash


def test_scoring_before_fit_or_for_unknown_investor_fails_loudly():
    model = LabelSimilarityRanker(COMPANIES)
    with pytest.raises(RuntimeError, match="before fit"):
        model.score(0, "inv", np.array(["sw_ca"]))
    model.fit(pd.DataFrame({"investor_permalink": ["inv"], "company_permalink": ["sw_ca"]}))
    with pytest.raises(KeyError, match="No training history"):
        model.score(0, "stranger", np.array(["sw_ca"]))


# Challenger: label_similarity_full (a missing label counts as a non-match). Written BEFORE the code.

from recommender.models.label_similarity import LabelSimilarityFullRanker  # noqa: E402


def _both():
    train = pd.DataFrame({"investor_permalink": ["inv"] * 3, "company_permalink": ["sw_ca", "sw_ca", "sw_ny"]})
    return LabelSimilarityRanker(COMPANIES).fit(train), LabelSimilarityFullRanker(COMPANIES).fit(train)


def test_challenger_pass_mark_1_fully_labelled_startups_score_exactly_the_same():
    old, new = _both()
    candidates = np.array(["sw_ca", "sw_ny", "bio_ca", "bio_ny"])
    assert np.array_equal(old.score(0, "inv", candidates), new.score(0, "inv", candidates))


def test_challenger_pass_mark_1_one_missing_label_divides_by_root_two():
    companies = pd.concat([COMPANIES, pd.DataFrame({"permalink": ["sw_only"], "category_code": ["software"],
                                                     "state_code": [None]})], ignore_index=True)
    train = pd.DataFrame({"investor_permalink": ["inv"] * 2, "company_permalink": ["sw_ca", "sw_ny"]})
    old = LabelSimilarityRanker(companies).fit(train).score(0, "inv", np.array(["sw_only"]))[0]
    new = LabelSimilarityFullRanker(companies).fit(train).score(0, "inv", np.array(["sw_only"]))[0]
    assert new == pytest.approx(old / np.sqrt(2), abs=1e-15)
    assert new < old


def test_challenger_no_labels_still_scores_zero():
    _, new = _both()
    assert list(new.score(0, "inv", np.array(["no_labels", "not_in_table"]))) == [0.0, 0.0]


def test_challenger_has_its_own_name():
    _, new = _both()
    assert new.name == "label_similarity_full"
