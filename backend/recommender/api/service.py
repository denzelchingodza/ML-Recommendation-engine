"""The engine behind the API: the champion's picks, explained, with a hindsight check.

Built once at startup, with the same data, draft board and ranking code as the
scoreboard, so the API can never drift from what the gate judged.

- The champion comes from champion.json (whatever the gate last crowned).
- Only models with an exact explainer can be served: no pick without an honest why.
- Hindsight: for investors in the 2012 test, each pick says whether they really
  backed that startup later. Evaluation you can see.
"""

import json
from pathlib import Path

import numpy as np

from recommender.data.build_dataset import build_dataset
from recommender.data.load_data import load_investments
from recommender.data.split_data import DEFAULT_MIN_HISTORY
from recommender.evaluation.draft_board import build_draft_board
from recommender.evaluation.scoreboard_core import rank_candidates
from recommender.explain.label_explanations import LabelExplainer
from recommender.models.label_similarity import LabelSimilarityFullRanker, LabelSimilarityRanker

CHAMPION_PATH = Path(__file__).resolve().parents[2] / "champion.json"
SERVABLE = {m.name: m for m in (LabelSimilarityRanker, LabelSimilarityFullRanker)}
MAX_K = 50


class UnknownInvestor(KeyError):
    """No investor with this id has enough history to get picks."""


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
        self.seen = history[history.map(len) >= DEFAULT_MIN_HISTORY].to_dict()
        self.later = self.dataset.test.groupby("investor_permalink")["company_permalink"].apply(set).to_dict()
        # Same tie-break order as the scoreboard: test investors first, in sorted order
        test_order = sorted(self.later)
        others = sorted(set(self.seen) - set(test_order))
        self.index = {inv: i for i, inv in enumerate(test_order + others)}
        names = investments.dropna(subset=["investor_permalink"]).drop_duplicates("investor_permalink")
        self.names = names.set_index("investor_permalink")["investor_name"].astype(str).to_dict()

    def search(self, text: str, limit: int) -> list[dict]:
        text = text.strip().lower()
        hits = [inv for inv in sorted(self.seen)
                if not text or text in inv.lower() or text in self.names.get(inv, "").lower()]
        hits.sort(key=lambda inv: -len(self.seen[inv]))  # busiest investors first
        return [{"id": inv, "name": self.names.get(inv, inv), "past_deals": len(self.seen[inv])} for inv in hits[:limit]]

    def picks(self, investor: str, k: int) -> dict:
        if investor not in self.seen:
            raise UnknownInvestor(investor)
        seen = self.seen[investor]
        candidates = self.board[~np.isin(self.board, list(seen))]
        index = self.index[investor]
        scores = self.model.score(index, investor, candidates)
        ranked = rank_candidates(candidates, scores, np.random.default_rng([self.seed, index]))[:k]
        later = self.later.get(investor)
        out = []
        for company in ranked:
            e = self.explainer.explain(investor, company)
            out.append({
                "company": company,
                "name": self.explainer._show(company),
                "score": e.score,
                "why": e.text,
                "by_chance": e.by_chance,
                "reasons": [{"label": r.column.removesuffix("_code"), "value": r.value, "part": r.part,
                             "past_deals": r.past_deals, "examples": list(r.examples)} for r in e.reasons],
                "backed_later": bool(later and company in later),
            })
        return {"investor": investor, "name": self.names.get(investor, investor), "past_deals": len(seen),
                "champion": self.champion, "cutoff": self.dataset.split_report.cutoff,
                "hindsight_available": later is not None, "picks": out}
