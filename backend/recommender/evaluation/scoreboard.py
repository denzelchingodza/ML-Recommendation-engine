"""The scoreboard: score a model on every test investor, then add confidence intervals.

Run it from the backend folder:
    python -m recommender.evaluation.scoreboard
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from recommender.data.build_dataset import Dataset, build_dataset
from recommender.data.load_data import load_investments
from recommender.evaluation.draft_board import build_draft_board, check_board_covers_test
from recommender.maths.confidence_intervals import (
    ConfidenceInterval,
    bootstrap_mean_ci,
    paired_bootstrap_diff_ci,
)
from recommender.maths.ranking_metrics import ndcg_at_k, recall_at_k
from recommender.models.baselines import MostPopularRanker, RandomRanker

DEFAULT_K = 10
SEGMENTS = ("overall", "known", "new")
METRICS = ("recall", "ndcg")


def rank_candidates(candidates: np.ndarray, scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Highest score first. Ties are broken RANDOMLY, never alphabetically."""
    tiebreak = rng.random(len(candidates))
    order = np.lexsort((tiebreak, -scores))  # last key sorts first: score, then tiebreak
    return candidates[order]


def score_model(model, dataset: Dataset, board: np.ndarray, k: int = DEFAULT_K, seed: int = 0) -> pd.DataFrame:
    """One row per test investor, with recall@k and NDCG@k for each segment.

    Segments: overall (all hidden deals), known (companies seen in training),
    new (companies never seen in training). A segment is NaN when the investor
    has no hidden deals of that kind, so it is left out, never counted as 0.
    """
    check_board_covers_test(board, dataset.test)
    train, test = dataset.train, dataset.test
    known_companies = set(train["company_permalink"])
    seen_by_investor = train.groupby("investor_permalink")["company_permalink"].apply(set)
    hidden_by_investor = test.groupby("investor_permalink")["company_permalink"].apply(set)

    rows = []
    for index, investor in enumerate(sorted(hidden_by_investor.index)):
        seen = seen_by_investor.get(investor, set())
        candidates = board[~np.isin(board, list(seen))]  # never recommend what they already backed
        scores = model.score(index, investor, candidates)
        if len(scores) != len(candidates):
            raise ValueError(f"{model.name} returned {len(scores)} scores for {len(candidates)} candidates")
        ranked = rank_candidates(candidates, scores, np.random.default_rng([seed, index]))

        hidden = hidden_by_investor[investor]
        segment_sets = {
            "overall": hidden,
            "known": hidden & known_companies,
            "new": hidden - known_companies,
        }
        row = {"investor": investor, "n_candidates": len(candidates)}
        for segment, relevant in segment_sets.items():
            row[f"n_hidden_{segment}"] = len(relevant)
            for metric, func in (("recall", recall_at_k), ("ndcg", ndcg_at_k)):
                row[f"{metric}_{segment}"] = func(ranked, relevant, k) if relevant else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class SegmentResult:
    metric: str
    segment: str
    interval: ConfidenceInterval


def summarise(per_investor: pd.DataFrame, n_resamples: int = 10_000, seed: int = 0) -> list[SegmentResult]:
    """Mean score with a 95% bootstrap interval, for every metric and segment."""
    results = []
    for segment in SEGMENTS:
        for metric in METRICS:
            values = per_investor[f"{metric}_{segment}"].dropna()
            results.append(SegmentResult(metric, segment, bootstrap_mean_ci(values, n_resamples, seed=seed)))
    return results


def compare(per_a: pd.DataFrame, per_b: pd.DataFrame, metric: str, segment: str,
            n_resamples: int = 10_000, seed: int = 0) -> ConfidenceInterval:
    """Paired interval for (b minus a), using only investors scored in both."""
    column = f"{metric}_{segment}"
    joined = per_a[["investor", column]].merge(per_b[["investor", column]], on="investor", suffixes=("_a", "_b"))
    joined = joined.dropna()
    return paired_bootstrap_diff_ci(joined[f"{column}_a"], joined[f"{column}_b"], n_resamples, seed=seed)


def expected_random_recall(per_investor: pd.DataFrame, k: int = DEFAULT_K) -> float:
    """What maths says random should score: picking k of n candidates catches k/n of hidden deals."""
    return float(np.mean(np.minimum(k, per_investor["n_candidates"]) / per_investor["n_candidates"]))


def run(k: int = DEFAULT_K, seed: int = 0) -> dict:
    dataset = build_dataset()
    board = build_draft_board(load_investments(), dataset.split_report.cutoff)
    results = {}
    for model in (RandomRanker(seed), MostPopularRanker()):
        model.fit(dataset.train)
        per_investor = score_model(model, dataset, board, k, seed)
        results[model.name] = {"per_investor": per_investor, "summary": summarise(per_investor, seed=seed)}
    return {"board_size": len(board), "k": k, "dataset": dataset, "results": results}


def format_report(output: dict) -> str:
    k, results = output["k"], output["results"]
    random_per = results["random"]["per_investor"]
    lines = [
        f"Phase 2 scoreboard (k = {k}, draft board = {output['board_size']:,} companies, "
        f"{len(random_per):,} test investors)",
        "",
    ]
    for name, result in results.items():
        lines.append(name)
        for r in result["summary"]:
            ci = r.interval
            lines.append(f"  {r.metric:>6}@{k} {r.segment:<8} {ci.mean:.4f}  (95% CI {ci.low:.4f} to {ci.high:.4f}, n={ci.n})")
        lines.append("")
    lines.append(f"Maths check: random recall@{k} should be about {expected_random_recall(random_per, k):.4f}")
    lines.append("")
    lines.append("most_popular minus random (paired, 95% CI)")
    for segment in SEGMENTS:
        diff = compare(random_per, results["most_popular"]["per_investor"], "ndcg", segment)
        verdict = "real win" if diff.low > 0 else ("real loss" if diff.high < 0 else "could be luck")
        lines.append(f"  ndcg@{k} {segment:<8} {diff.mean:+.4f}  ({diff.low:+.4f} to {diff.high:+.4f})  {verdict}")
    return "\n".join(lines)


def main() -> None:
    print(format_report(run()))


if __name__ == "__main__":
    main()
