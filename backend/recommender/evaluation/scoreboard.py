"""The scoreboard: score a model on every test investor, then add confidence intervals.

Run it from the backend folder:
    python -m recommender.evaluation.scoreboard
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from recommender.data.build_dataset import Dataset, build_dataset
from recommender.data.load_data import load_investments
from recommender.evaluation.draft_board import build_draft_board
from recommender.evaluation.scoreboard_core import DEFAULT_K, rank_candidates, score_model  # noqa: F401 (re-exported)
from recommender.evaluation.tuning import select_blend_weight, select_settings
from recommender.maths.confidence_intervals import (
    ConfidenceInterval,
    bootstrap_mean_ci,
    paired_bootstrap_diff_ci,
)
from recommender.models.baselines import MostPopularRanker, RandomRanker
from recommender.models.label_similarity import LabelSimilarityRanker
from recommender.models.hybrid import HybridRanker
from recommender.models.matrix_factorisation import MatrixFactorisationRanker

SEGMENTS = ("overall", "known", "new")
METRICS = ("recall", "ndcg")
COMPARISONS = (
    ("random", "most_popular"),
    ("random", "label_similarity"),
    ("most_popular", "label_similarity"),
    ("label_similarity", "matrix_factorisation"),
    ("label_similarity", "hybrid"),
    ("matrix_factorisation", "hybrid"),
)


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
    settings, tuning = select_settings(dataset.train, dataset.companies, dataset.split_report.cutoff, k=k, seed=seed)
    weight, blend_tuning = select_blend_weight(dataset.train, dataset.companies, dataset.split_report.cutoff,
                                               settings, k=k, seed=seed)
    results = {}
    models = (
        RandomRanker(seed),
        MostPopularRanker(),
        LabelSimilarityRanker(dataset.companies),
        MatrixFactorisationRanker(dataset.companies, seed=seed, **settings),
        HybridRanker(MatrixFactorisationRanker(dataset.companies, seed=seed, **settings),
                     LabelSimilarityRanker(dataset.companies), weight),
    )
    for model in models:
        model.fit(dataset.train)
        per_investor = score_model(model, dataset, board, k, seed)
        results[model.name] = {"per_investor": per_investor, "summary": summarise(per_investor, seed=seed)}
    return {"board_size": len(board), "k": k, "dataset": dataset, "results": results,
            "mf_settings": settings, "mf_tuning": tuning, "mf_model": models[3],
            "blend_weight": weight, "blend_tuning": blend_tuning}


def format_report(output: dict) -> str:
    k, results = output["k"], output["results"]
    random_per = results["random"]["per_investor"]
    lines = [
        f"Scoreboard (k = {k}, draft board = {output['board_size']:,} companies, "
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
    if "mf_tuning" in output:
        lines.append("")
        lines.append("Matrix factorisation settings, chosen on the practice split (deals before 2012 only):")
        for settings, score in output["mf_tuning"]:
            mark = "  <- chosen" if settings == output["mf_settings"] else ""
            lines.append(f"  {settings}: practice ndcg@{k} {score:.4f}{mark}")
    if "blend_tuning" in output:
        lines.append("")
        lines.append("Hybrid weight on matrix factorisation (rest on label similarity), chosen on the practice split:")
        for w, score in output["blend_tuning"]:
            mark = "  <- chosen" if w == output["blend_weight"] else ""
            lines.append(f"  w = {w:.1f}: practice ndcg@{k} {score:.4f}{mark}")
    lines.append("")
    for base, challenger in COMPARISONS:
        if base not in results or challenger not in results:
            continue
        lines.append(f"{challenger} minus {base} (paired, 95% CI)")
        for segment in SEGMENTS:
            diff = compare(results[base]["per_investor"], results[challenger]["per_investor"], "ndcg", segment)
            verdict = "real win" if diff.low > 0 else ("real loss" if diff.high < 0 else "could be luck")
            lines.append(f"  ndcg@{k} {segment:<8} {diff.mean:+.4f}  ({diff.low:+.4f} to {diff.high:+.4f})  {verdict}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    print(format_report(run()))


if __name__ == "__main__":
    main()
