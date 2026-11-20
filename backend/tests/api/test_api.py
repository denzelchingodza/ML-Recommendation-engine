"""Phase 8a pass marks for the API. Written BEFORE the code.

1 Same as the scoreboard: picks and scores match score_model for the champion exactly.
2 Honest whys: every pick's explanation parts add up to its score.
3 Follows the gate: the API serves whatever champion.json names.
4 Clean errors: unknown investor 404, bad k 422, never a 500 for bad input.
5 Fast: a request for picks takes under 1 second after startup.
"""

import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from recommender.api.app import create_app
from recommender.api.service import RecommendationService
from recommender.evaluation.scoreboard_core import score_model


@pytest.fixture(scope="module")
def service():
    return RecommendationService()


@pytest.fixture(scope="module")
def client(service):
    with TestClient(create_app(service)) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_champion_comes_from_champion_json(client):
    body = client.get("/champion").json()
    on_disk = json.loads(open("champion.json").read())
    assert body["model"] == on_disk["champion"] and body["version"] == on_disk["version"]


def test_pass_mark_1_same_picks_and_scores_as_the_scoreboard(service, client):
    per = score_model(service.model, service.dataset, service.board, k=10, seed=0)
    for _, row in per.iterrows():
        body = client.get("/picks", params={"investor": row["investor"], "k": 10}).json()
        assert tuple(p["company"] for p in body["picks"]) == row["top_k"], row["investor"]
        scores = service.model.score(0, row["investor"], np.array(row["top_k"]))
        assert [p["score"] for p in body["picks"]] == pytest.approx(list(scores), abs=1e-12)


def test_pass_mark_2_every_why_adds_up(service, client):
    investor = service.dataset.test["investor_permalink"].iloc[0]
    for pick in client.get("/picks", params={"investor": investor, "k": 50}).json()["picks"]:
        assert pick["why"]
        assert sum(r["part"] for r in pick["reasons"]) == pytest.approx(pick["score"], abs=1e-12)


def test_hindsight_marks_only_real_later_deals(service, client):
    investor = service.dataset.test["investor_permalink"].iloc[0]
    later = set(service.dataset.test.loc[service.dataset.test["investor_permalink"] == investor, "company_permalink"])
    body = client.get("/picks", params={"investor": investor, "k": 50}).json()
    for pick in body["picks"]:
        assert pick["backed_later"] == (pick["company"] in later)
    assert body["hindsight_available"] is True


def test_pass_mark_3_follows_champion_json(tmp_path):
    path = tmp_path / "champion.json"
    path.write_text(json.dumps({"champion": "label_similarity", "version": 1}))
    other = RecommendationService(champion_path=path)
    assert other.champion == {"model": "label_similarity", "version": 1}
    assert other.model.name == "label_similarity"


def test_pass_mark_3_refuses_a_champion_it_cannot_explain(tmp_path):
    path = tmp_path / "champion.json"
    path.write_text(json.dumps({"champion": "hybrid", "version": 3}))
    with pytest.raises(ValueError, match="explain"):
        RecommendationService(champion_path=path)


@pytest.mark.parametrize("params,status", [
    ({"investor": "/financial-organization/does-not-exist"}, 404),
    ({}, 422),
    ({"investor": ""}, 422),
    ({"investor": "x", "k": 0}, 422),
    ({"investor": "x", "k": 51}, 422),
    ({"investor": "x", "k": "ten"}, 422),
])
def test_pass_mark_4_bad_input_gets_a_clear_error_never_a_500(client, params, status):
    response = client.get("/picks", params=params)
    assert response.status_code == status
    assert "detail" in response.json()


def test_pass_mark_4_unknown_investor_message_says_what_to_do(client):
    detail = client.get("/picks", params={"investor": "/person/nobody"}).json()["detail"]
    assert "/investors" in detail


def test_search_finds_investors_and_validates_input(client):
    found = client.get("/investors", params={"search": "sequoia", "limit": 5}).json()["investors"]
    assert found and all("sequoia" in (i["name"] + i["id"]).lower() for i in found)
    # Phase 8b: search now also lists investors with too little history, flagged, so the page can say so
    assert all(i["enough_history"] == (i["past_deals"] >= 5) for i in found)
    assert client.get("/investors", params={"limit": 0}).status_code == 422
    assert client.get("/investors", params={"search": "zzzz-no-such-investor"}).json()["investors"] == []


def test_pass_mark_5_picks_come_back_in_under_a_second(service, client):
    investor = service.dataset.test["investor_permalink"].iloc[0]
    client.get("/picks", params={"investor": investor})  # warm up
    start = time.perf_counter()
    client.get("/picks", params={"investor": investor, "k": 10})
    assert time.perf_counter() - start < 1.0
