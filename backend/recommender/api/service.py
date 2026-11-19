"""The engine behind the API: the champion's picks, explained, with a hindsight check.

Built once at startup, with the same data, draft board and ranking code as the
scoreboard, so the API can never drift from what the gate judged.

- The champion comes from champion.json (whatever the gate last crowned).
- Only models with an exact explainer can be served: no pick without an honest why.
- Hindsight: for investors in the 2012 test, each pick says whether they really
  backed that startup later. Evaluation you can see.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from recommender.data.build_dataset import build_dataset
from recommender.data.load_data import load_investments
from recommender.data.split_data import DEFAULT_MIN_HISTORY
from recommender.evaluation.draft_board import build_draft_board
from recommender.evaluation.scoreboard_core import rank_candidates
from recommender.api.places import readable
from recommender.explain.label_explanations import LabelExplainer
from recommender.models.label_similarity import LabelSimilarityFullRanker, LabelSimilarityRanker

CHAMPION_PATH = Path(__file__).resolve().parents[2] / "champion.json"
SERVABLE = {m.name: m for m in (LabelSimilarityRanker, LabelSimilarityFullRanker)}
MAX_K = 50


class UnknownInvestor(KeyError):
    """No investor with this id made any investment before the cutoff."""


class NotEnoughHistory(ValueError):
    """The investor exists but has too few past investments to learn their habits from."""

    def __init__(self, investor: str, past_deals: int):
        super().__init__(investor)
        self.investor, self.past_deals, self.needed = investor, past_deals, DEFAULT_MIN_HISTORY


def match_state(labels_on_record: int, shared: int, by_chance: bool) -> str:
    """The plain state a page shows instead of a score.

    close    : the startup has both labels on record and shares both with the investor's past
    partial  : it has both on record but shares only one
    possible : it is missing a label on record (so it cannot be fully checked), or shares nothing
    """
    if by_chance or shared == 0 or labels_on_record < 2:
        return "possible"
    return "close" if shared >= 2 else "partial"


def read_champion(path: Path) -> dict:
    try:
        entry = json.loads(Path(path).read_text())
        return {"model": entry["champion"], "version": int(entry["version"])}
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Could not read the champion from {path}: {error}") from error


class RecommendationService:
    def __init__(self, champion_path: Path = CHAMPION_PATH, seed: int = 0):
        self.champion = read_champion(champion_path)
        if self.champion["model"] not in SERVABLE:
            raise ValueError(f"The champion {self.champion['model']!r} has no exact explainer, so the API cannot "
                             f"explain its picks. Servable models: {sorted(SERVABLE)}")
        self.seed = seed
        self.dataset = build_dataset()
        investments = load_investments()
        self.board = build_draft_board(investments, self.dataset.split_report.cutoff)
        train = self.dataset.train
        self.model = SERVABLE[self.champion["model"]](self.dataset.companies).fit(train)
        self.explainer = LabelExplainer(self.model, train, self.dataset.companies)

        history = train.groupby("investor_permalink")["company_permalink"].apply(set)
        self.history = history.to_dict()                                     # everyone with any past investment
        self.seen = history[history.map(len) >= DEFAULT_MIN_HISTORY].to_dict()  # enough to learn habits from
        labels = self.dataset.companies.set_index("permalink")[["category_code", "state_code"]]
        self.labels = {c: {k: v for k, v in row.items() if isinstance(v, str)} for c, row in labels.iterrows()}
        self.later = self.dataset.test.groupby("investor_permalink")["company_permalink"].apply(set).to_dict()
        # Same tie-break order as the scoreboard: test investors first, in sorted order
        test_order = sorted(self.later)
        others = sorted(set(self.seen) - set(test_order))
        self.index = {inv: i for i, inv in enumerate(test_order + others)}
        names = investments.dropna(subset=["investor_permalink"]).drop_duplicates("investor_permalink")
        self.names = names.set_index("investor_permalink")["investor_name"].astype(str).to_dict()

    def search(self, text: str, limit: int) -> list[dict]:
        text = text.strip().lower()
        hits = [inv for inv in sorted(self.history)
                if not text or text in inv.lower() or text in self.names.get(inv, "").lower()]
        hits.sort(key=lambda inv: -len(self.history[inv]))  # busiest investors first
        return [{"id": inv, "name": self.names.get(inv, inv), "past_deals": len(self.history[inv]),
                 "enough_history": inv in self.seen} for inv in hits[:limit]]

    def _check(self, investor: str) -> None:
        if investor not in self.history:
            raise UnknownInvestor(investor)
        if investor not in self.seen:
            raise NotEnoughHistory(investor, len(self.history[investor]))

    def profile(self, investor: str) -> dict:
        """The investor's habits in plain terms, for the Overview tab. Answers even with too little history."""
        if investor not in self.history:
            raise UnknownInvestor(investor)
        past = self.history[investor]
        states = Counter(self.labels.get(c, {}).get("state_code") for c in past)
        sectors = Counter(self.labels.get(c, {}).get("category_code") for c in past)
        states.pop(None, None), sectors.pop(None, None)
        top = lambda counts: min(counts.items(), key=lambda kv: (-kv[1], kv[0])) if counts else None  # noqa: E731
        st, sec = top(states), top(sectors)
        later = self.later.get(investor)
        return {"id": investor, "name": self.names.get(investor, investor), "past_deals": len(past),
                "enough_history": investor in self.seen, "needed": DEFAULT_MIN_HISTORY,
                "cutoff": self.dataset.split_report.cutoff,
                "top_state": {"code": st[0], "name": readable("state", st[0]), "count": st[1]} if st else None,
                "top_sector": {"name": readable("category", sec[0]), "count": sec[1]} if sec else None,
                "later_investments": len(later) if later is not None else None}

    def picks(self, investor: str, k: int) -> dict:
        self._check(investor)
        seen = self.seen[investor]
        candidates = self.board[~np.isin(self.board, list(seen))]
        index = self.index[investor]
        scores = self.model.score(index, investor, candidates)
        ranked = rank_candidates(candidates, scores, np.random.default_rng([self.seed, index]))[:k]
        score_of = dict(zip(candidates, scores))
        later = self.later.get(investor)
        out = []
        for company in ranked:
            e = self.explainer.explain(investor, company)
            on_record = len(self.labels.get(company, {}))
            out.append({
                "company": company,
                "name": self.explainer._show(company),
                "score": e.score,
                "why": e.text,
                "by_chance": e.by_chance,
                "match": match_state(on_record, len(e.reasons), e.by_chance),
                "equally_strong": int(np.isclose(scores, score_of[company], rtol=0, atol=1e-12).sum()),
                "reasons": [{"label": r.column.removesuffix("_code"), "value": r.value,
                             "value_name": readable(r.column.removesuffix("_code"), r.value), "part": r.part,
                             "past_deals": r.past_deals, "examples": list(r.examples),
                             "example_names": [self.explainer._show(c) for c in r.examples]} for r in e.reasons],
                "backed_later": bool(later and company in later),
            })
        return {"investor": investor, "name": self.names.get(investor, investor), "past_deals": len(seen),
                "champion": self.champion, "cutoff": self.dataset.split_report.cutoff,
                "hindsight_available": later is not None, "picks": out}
