"""Phase 8b API additions the demo page needs. Written BEFORE the code.

1 Each pick has a plain match state: close, partial or possible.
2 Each pick says how many startups on the board match exactly as well (equal matches).
3 "Not enough history" is its own answer, separate from "unknown investor".
Plus: an investor profile (their habits in plain terms) for the Overview tab.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from recommender.api.app import create_app
from recommender.api.places import US_STATE_NAMES
from recommender.api.service import RecommendationService, match_state

AUSTIN = "/financial-organization/austin-ventures"
ACT = "/financial-organization/act-venture-capital"


@pytest.fixture(scope="module")
def service():
    return RecommendationService()


@pytest.fixture(scope="module")
def client(service):
    with TestClient(create_app(service)) as c:
        yield c


# 1 Match states


@pytest.mark.parametrize("labels_on_record,shared,by_chance,expected", [
    (2, 2, False, "close"),     # same sector and same state
    (2, 1, False, "partial"),   # only one of the two lines up
    (1, 1, False, "possible"),  # the startup is missing a label, so we cannot fully check it
    (0, 0, True, "possible"),   # nothing shared at all
    (2, 0, True, "possible"),
])
def test_match_state_rules(labels_on_record, shared, by_chance, expected):
    assert match_state(labels_on_record, shared, by_chance) == expected


def test_every_pick_has_a_match_state(client):
    picks = client.get("/picks", params={"investor": AUSTIN, "k": 50}).json()["picks"]
    assert {p["match"] for p in picks} <= {"close", "partial", "possible"}
    assert all(p["match"] == "close" for p in picks[:10])  # Austin's top 10: software in Texas, both shared


# 2 Equal matches


def test_equal_matches_count_the_whole_board(service, client):
    body = client.get("/picks", params={"investor": AUSTIN, "k": 10}).json()
    seen = service.seen[AUSTIN]
    candidates = service.board[~np.isin(service.board, list(seen))]
    scores = service.model.score(0, AUSTIN, candidates)
    for pick in body["picks"]:
        expected = int(np.isclose(scores, pick["score"], rtol=0, atol=1e-12).sum())
        assert pick["equally_strong"] == expected
    assert body["picks"][0]["equally_strong"] == 27   # measured for the design preview



def test_searched_counts_every_startup_the_model_scored(service, client):
    body = client.get("/picks", params={"investor": AUSTIN, "k": 10}).json()
    seen = service.seen[AUSTIN]
    expected = int((~np.isin(service.board, list(seen))).sum())
    assert body["searched"] == expected
    assert body["searched"] < len(service.board)   # startups Austin already backed are left out


# 3 Not enough history


def test_not_enough_history_is_its_own_answer(client):
    response = client.get("/picks", params={"investor": ACT})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "not_enough_history"
    assert detail["past_deals"] == 3 and detail["needed"] == 5
    assert "5" in detail["message"] and "3" in detail["message"]


def test_unknown_investor_is_still_404(client):
    assert client.get("/picks", params={"investor": "/person/nobody"}).status_code == 404


def test_search_includes_investors_with_too_little_history(client):
    found = client.get("/investors", params={"search": "act venture capital"}).json()["investors"]
    act = next(i for i in found if i["id"] == ACT)
    assert act["past_deals"] == 3 and act["enough_history"] is False


# Profile for the Overview tab


def test_profile_describes_their_habits(client):
    body = client.get("/investor", params={"investor": AUSTIN}).json()
    assert body["name"] == "Austin Ventures" and body["past_deals"] == 85 and body["enough_history"] is True
    assert body["top_state"] == {"code": "TX", "name": "Texas", "count": 59}
    assert body["top_sector"] == {"name": "software", "count": 19}
    assert body["later_investments"] == 19
    assert body["cutoff"] == "2012-01-01"


def test_profile_for_too_little_history_still_answers(client):
    body = client.get("/investor", params={"investor": ACT}).json()
    assert body["enough_history"] is False and body["past_deals"] == 3 and body["needed"] == 5


def test_profile_unknown_investor_is_404(client):
    response = client.get("/investor", params={"investor": "/person/nobody"})
    assert response.status_code == 404 and "/investors" in response.json()["detail"]


# Readable names for the page


def test_reasons_carry_readable_names(client):
    pick = client.get("/picks", params={"investor": AUSTIN, "k": 1}).json()["picks"][0]
    by_label = {r["label"]: r for r in pick["reasons"]}
    assert by_label["state"]["value_name"] == "Texas"
    assert by_label["category"]["value_name"] == "software"
    assert by_label["state"]["example_names"] == ["Adometry", "Alereon"]


def test_every_state_code_in_the_data_has_a_name(service):
    codes = set(service.dataset.companies["state_code"].dropna())
    assert codes <= set(US_STATE_NAMES), sorted(codes - set(US_STATE_NAMES))
