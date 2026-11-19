"""Explaining label similarity picks, on the hand-made example from the plan. Written BEFORE the code."""

import numpy as np
import pandas as pd
import pytest

from recommender.explain.label_explanations import LabelExplainer
from recommender.models.label_similarity import LabelSimilarityRanker

COMPANIES = pd.DataFrame(
    {
        "permalink": ["acme", "bolt", "cove", "sw_ca", "bio_tx", "sw_none", "blank"],
        "category_code": ["software", "software", "fintech", "software", "biotech", "software", None],
        "state_code": ["NY", "NY", "CA", "CA", "TX", None, None],
    }
)
# investor backed two software startups in NY and one fintech in CA: profile [software 2, fintech 1, NY 2, CA 1]
TRAIN = pd.DataFrame({"investor_permalink": ["inv"] * 3, "company_permalink": ["acme", "bolt", "cove"]})


@pytest.fixture
def explainer():
    model = LabelSimilarityRanker(COMPANIES).fit(TRAIN)
    return LabelExplainer(model, TRAIN), model


def test_parts_match_the_worked_example(explainer):
    ex, _ = explainer
    e = ex.explain("inv", "sw_ca")
    parts = {r.value: r.part for r in e.reasons}
    assert parts["software"] == pytest.approx(2 / (np.sqrt(10) * np.sqrt(2)))  # 0.45
    assert parts["CA"] == pytest.approx(1 / (np.sqrt(10) * np.sqrt(2)))        # 0.22
    assert e.score == pytest.approx(3 / (np.sqrt(10) * np.sqrt(2)))           # 0.67


def test_parts_add_up_to_the_model_score(explainer):
    ex, model = explainer
    for company in ["sw_ca", "bio_tx", "sw_none", "blank", "acme"]:
        e = ex.explain("inv", company)
        model_score = model.score(0, "inv", np.array([company]))[0]
        assert sum(r.part for r in e.reasons) == pytest.approx(model_score, abs=1e-12)
        assert e.score == pytest.approx(model_score, abs=1e-12)


def test_biggest_reason_comes_first(explainer):
    ex, _ = explainer
    e = ex.explain("inv", "sw_ca")
    assert [r.value for r in e.reasons] == ["software", "CA"]


def test_evidence_is_real_past_deals_sharing_the_label(explainer):
    ex, _ = explainer
    e = ex.explain("inv", "sw_ca")
    by_value = {r.value: r for r in e.reasons}
    assert by_value["software"].past_deals == 2 and by_value["software"].examples == ("acme", "bolt")
    assert by_value["CA"].past_deals == 1 and by_value["CA"].examples == ("cove",)


def test_no_shared_label_is_called_chance_not_given_a_fake_reason(explainer):
    ex, _ = explainer
    for company in ["bio_tx", "blank", "not_in_table"]:
        e = ex.explain("inv", company)
        assert e.by_chance and e.reasons == () and e.score == 0.0
        assert "by chance" in e.text


def test_one_shared_label_only(explainer):
    ex, _ = explainer
    e = ex.explain("inv", "sw_none")
    assert [r.value for r in e.reasons] == ["software"]
    assert not e.by_chance


def test_text_reads_plainly(explainer):
    ex, _ = explainer
    text = ex.explain("inv", "sw_ca").text
    assert text.startswith("Software startup in CA.")
    assert "software deals (2, e.g. acme, bolt)" in text
    assert "CA deals (1, e.g. cove)" in text
    assert "about 67% of the match" in text  # software's share: 0.45 / 0.67
    assert "\u2014" not in text                # no em dashes


def test_unknown_investor_fails_loudly(explainer):
    ex, _ = explainer
    with pytest.raises(KeyError):
        ex.explain("stranger", "sw_ca")


def test_state_codes_keep_their_capitals():
    # an investor whose biggest reason is a state: 2 CA deals (cove, ca_bio), 1 software deal (acme)
    extra = pd.DataFrame({"permalink": ["ca_bio"], "category_code": ["biotech"], "state_code": ["CA"]})
    train = pd.DataFrame({"investor_permalink": ["ca_fan"] * 3, "company_permalink": ["cove", "ca_bio", "acme"]})
    model = LabelSimilarityRanker(pd.concat([COMPANIES, extra], ignore_index=True)).fit(train)
    text = LabelExplainer(model, train).explain("ca_fan", "sw_ca").text
    assert "CA counts for about" in text and "Ca counts" not in text


def test_real_names_are_shown_and_junk_names_fall_back_to_the_address():
    companies = COMPANIES.assign(name=["Acme Inc", "#NAME?", None, "SW CA", "Bio TX", "SW", "Blank"],
                                 permalink=["/company/" + p for p in COMPANIES["permalink"]])
    train = pd.DataFrame({"investor_permalink": ["inv"] * 3,
                          "company_permalink": ["/company/acme", "/company/bolt", "/company/cove"]})
    model = LabelSimilarityRanker(companies).fit(train)
    e = LabelExplainer(model, train, companies).explain("inv", "/company/sw_ca")
    assert "(2, e.g. Acme Inc, bolt)" in e.text      # "#NAME?" is junk, so the address is shown
    assert "(1, e.g. cove)" in e.text                 # no name at all, so the address is shown
    assert e.reasons[0].examples == ("/company/acme", "/company/bolt")  # evidence keeps the real ids


# Challenger: the same exact explanations for label_similarity_full. Written BEFORE the code.

from recommender.models.label_similarity import LabelSimilarityFullRanker  # noqa: E402


def test_challenger_parts_add_up_to_its_score():
    model = LabelSimilarityFullRanker(COMPANIES).fit(TRAIN)
    ex = LabelExplainer(model, TRAIN)
    for company in ["sw_ca", "bio_tx", "sw_none", "blank", "acme"]:
        e = ex.explain("inv", company)
        assert e.score == pytest.approx(model.score(0, "inv", np.array([company]))[0], abs=1e-12)
        assert sum(r.part for r in e.reasons) == pytest.approx(e.score, abs=1e-12)


def test_challenger_says_a_missing_label_counted_as_no_match():
    model = LabelSimilarityFullRanker(COMPANIES).fit(TRAIN)
    text = LabelExplainer(model, TRAIN).explain("inv", "sw_none").text
    assert "No state on record, which counts as no match." in text


def test_champion_does_not_claim_that():
    text = LabelExplainer(LabelSimilarityRanker(COMPANIES).fit(TRAIN), TRAIN).explain("inv", "sw_none").text
    assert "on record" not in text
