"""Phase 5b: the multi-season scoreboard. One game cannot prove a small edge; a series can.

Each season, with cutoff C:
    learn from:   every deal before C
    tested on:    deals in the 12 months from C (the test window)
    draft board:  companies that raised money inside that window
    tuning:       the SAME recipe as the main scoreboard, practised on the year
                  before C (learn before C minus 1 year, check on that year)

Then the seasons are combined. One investor can play in several seasons, so the
combined interval resamples whole investors (cluster bootstrap), never single rows.

The 2012 season overlaps the test we already looked at while choosing models,
so the verdict is also reported on the fresh seasons (2009 to 2011) alone.

Run it from the backend folder:
    python -m recommender.evaluation.seasons
"""

from functools import lru_cache
from types import SimpleNamespace

import pandas as pd

from recommender.data.clean_data import clean_investments
from recommender.data.load_data import load_companies, load_investments
from recommender.data.split_data import DEFAULT_MIN_HISTORY, temporal_split
from recommender.evaluation.draft_board import build_draft_board
from recommender.evaluation.scoreboard_core import DEFAULT_K, exposure, score_model
from recommender.evaluation.tuning import select_blend_weight, select_exploration, select_settings
from recommender.maths.confidence_intervals import ConfidenceInterval, cluster_bootstrap_mean_ci
from recommender.models.baselines import MostPopularRanker, RandomRanker
from recommender.models.hybrid import HybridRanker
from recommender.models.label_similarity import LabelSimilarityRanker
from recommender.models.matrix_factorisation import MatrixFactorisationRanker
from recommender.models.thompson import ThompsonRanker

SEASON_CUTOFFS = ("2009-01-01", "2010-01-01", "2011-01-01", "2012-01-01")
FRESH_SEASONS = SEASON_CUTOFFS[:3]  # never looked at while choosing models
SEGMENTS = ("overall", "known", "new")
MODEL_NAMES = ("random", "most_popular", "label_similarity", "matrix_factorisation", "hybrid", "thompson_hybrid")
COMPARISONS = (
    ("random", "most_popular"),
    ("label_similarity", "matrix_factorisation"),
    ("label_similarity", "hybrid"),
    ("label_similarity", "thompson_hybrid"),
    ("hybrid", "thompson_hybrid"),
)


def _shift(date: str, years: int) -> str:
    return (pd.Timestamp(date) + pd.DateOffset(years=years)).strftime("%Y-%m-%d")


@lru_cache(maxsize=1)
def _raw():
    """Load and clean once; every season cuts its own split from the same cleaned deals."""
    investments = load_investments()
    pairs, _ = clean_investments(investments)
    return investments, load_companies(), pairs


def _first_deal_from(investments: pd.DataFrame, cutoff: str, board) -> pd.Series:
    """For each board company: the date of its first deal on or after the cutoff.

    Worked out separately from the board itself, so the leak test is a real check.
    """
    dates = pd.to_datetime(investments["funded_at"], format="%Y-%m-%d", errors="coerce")
    after = investments.assign(date=dates)[dates >= pd.Timestamp(cutoff)]
    return after.groupby("company_permalink")["date"].min().reindex(board)


@lru_cache(maxsize=None)
def run_season(cutoff: str, k: int = DEFAULT_K, seed: int = 0) -> dict:
    """One season, scored once per test run and then reused (the gate and the season tests share it).

    Treat the result as read-only. run_season.__wrapped__ recomputes from scratch.
    """
    investments, companies, pairs = _raw()
    end, practice_cutoff = _shift(cutoff, 1), _shift(cutoff, -1)

    train, test, report = temporal_split(pairs, cutoff, DEFAULT_MIN_HISTORY, end=end)
    dataset = SimpleNamespace(train=train, test=test, companies=companies, split_report=report)
    board = build_draft_board(investments, cutoff, end)

    tune = dict(k=k, seed=seed, practice_cutoff=practice_cutoff)
    settings, _ = select_settings(train, companies, cutoff, **tune)
    weight, _ = select_blend_weight(train, companies, cutoff, settings, **tune)
    strength, _, fair_found = select_exploration(train, companies, cutoff, settings, weight, **tune)

    def hybrid():
        return HybridRanker(MatrixFactorisationRanker(companies, seed=seed, **settings),
                            LabelSimilarityRanker(companies), weight)

    models = (
        RandomRanker(seed),
        MostPopularRanker(),
        LabelSimilarityRanker(companies),
        MatrixFactorisationRanker(companies, seed=seed, **settings),
        hybrid(),
        ThompsonRanker(hybrid(), strength, seed),
    )
    if tuple(m.name for m in models) != MODEL_NAMES:
        raise RuntimeError("MODEL_NAMES is out of date with the models scored in a season")
    known = set(train["company_permalink"])
    results = {}
    for model in models:
        model.fit(train)
        per_investor = score_model(model, dataset, board, k, seed)
        results[model.name] = {"per_investor": per_investor, "exposure": exposure(per_investor, board, known)}

    return {"cutoff": cutoff, "end": end, "practice_cutoff": practice_cutoff, "dataset": dataset,
            "board": board, "board_dates": _first_deal_from(investments, cutoff, board),
            "settings": settings, "weight": weight, "strength": strength, "ts_fair_found": fair_found,
            "results": results}


def run(cutoffs=SEASON_CUTOFFS, k: int = DEFAULT_K, seed: int = 0) -> dict:
    return {"k": k, "seasons": {c: run_season(c, k, seed) for c in cutoffs}}


def _paired_rows(output: dict, base: str, challenger: str, metric: str, segment: str, cutoffs) -> pd.DataFrame:
    column = f"{metric}_{segment}"
    frames = []
    for cutoff in cutoffs:
        results = output["seasons"][cutoff]["results"]
        a = results[base]["per_investor"][["investor", column]]
        b = results[challenger]["per_investor"][["investor", column]]
        joined = a.merge(b, on="investor", suffixes=("_a", "_b")).dropna()
        frames.append(pd.DataFrame({"investor": joined["investor"],
                                    "diff": joined[f"{column}_b"] - joined[f"{column}_a"]}))
    return pd.concat(frames, ignore_index=True)


def combined_diff(output: dict, base: str, challenger: str, metric: str = "ndcg", segment: str = "overall",
                  cutoffs=None, n_resamples: int = 10_000, seed: int = 0) -> ConfidenceInterval:
    """Interval for (challenger minus base) over the chosen seasons, resampling whole investors."""
    rows = _paired_rows(output, base, challenger, metric, segment, cutoffs or tuple(output["seasons"]))
    return cluster_bootstrap_mean_ci(rows["diff"], rows["investor"], n_resamples, seed=seed)


def _verdict(ci: ConfidenceInterval) -> str:
    return "real win" if ci.low > 0 else ("real loss" if ci.high < 0 else "could be luck")


def format_report(output: dict) -> str:
    k, seasons = output["k"], output["seasons"]
    lines = [f"Multi-season scoreboard (k = {k}): each season learns before its cutoff, tested on the next 12 months", ""]
    for cutoff, s in seasons.items():
        per = s["results"]["random"]["per_investor"]
        lines.append(f"Season {cutoff[:4]}  ({cutoff} to {s['end']}, practice year from {s['practice_cutoff']})")
        lines.append(f"  {len(per):,} test investors, board of {len(s['board']):,} companies")
        lines.append(f"  tuned: MF {s['settings']}, hybrid weight {s['weight']:.1f}, Thompson strength {s['strength']}"
                     f"{'' if s['ts_fair_found'] else ' (no fair strength on practice)'}")
        for name, r in s["results"].items():
            ex = r["exposure"]
            lines.append(f"  {name:<22} ndcg@{k} {r['per_investor']['ndcg_overall'].mean():.4f}   "
                         f"new share {ex['new_share_of_slots']:.0%} ({ex['new_share_of_board']:.0%})")
        lines.append("")
    for base, challenger in COMPARISONS:
        lines.append(f"{challenger} minus {base}, ndcg@{k} (paired, 95% CI)")
        for cutoff in seasons:
            ci = combined_diff(output, base, challenger, cutoffs=(cutoff,))
            lines.append(f"  season {cutoff[:4]:<14} {ci.mean:+.4f}  ({ci.low:+.4f} to {ci.high:+.4f})  {_verdict(ci)}")
        groups = [("all seasons", tuple(seasons))]
        fresh = tuple(c for c in FRESH_SEASONS if c in seasons)
        if fresh and len(fresh) < len(seasons):
            groups.append(("fresh seasons", fresh))
        for label, cutoffs in groups:
            for segment in SEGMENTS:
                ci = combined_diff(output, base, challenger, segment=segment, cutoffs=cutoffs)
                lines.append(f"  {label + ', ' + segment:<20} {ci.mean:+.4f}  ({ci.low:+.4f} to {ci.high:+.4f})  "
                             f"{_verdict(ci)}  n={ci.n}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    print(format_report(run()))


if __name__ == "__main__":
    main()
