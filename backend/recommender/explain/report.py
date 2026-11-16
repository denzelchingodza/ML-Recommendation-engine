"""Explain every top-10 pick the starter makes on the main scoreboard (2012 cutoff).

Checks each explanation against the model (the parts must add up to the score)
and reports how often a pick was shown by chance.

Run it from the backend folder:
    python -m recommender.explain.report
"""

import numpy as np

from recommender.data.build_dataset import build_dataset
from recommender.data.load_data import load_investments
from recommender.evaluation.draft_board import build_draft_board
from recommender.evaluation.scoreboard_core import DEFAULT_K, score_model
from recommender.explain.label_explanations import LabelExplainer
from recommender.models.label_similarity import LABEL_COLUMNS, LabelSimilarityRanker


def run(k: int = DEFAULT_K, seed: int = 0) -> dict:
    dataset = build_dataset()
    board = build_draft_board(load_investments(), dataset.split_report.cutoff)
    model = LabelSimilarityRanker(dataset.companies).fit(dataset.train)
    per_investor = score_model(model, dataset, board, k, seed)   # rows are in investor_index order
    explainer = LabelExplainer(model, dataset.train, dataset.companies)

    explanations, max_error = [], 0.0
    for index, row in per_investor.iterrows():
        investor = row["investor"]
        for company in row["top_k"]:
            e = explainer.explain(investor, company)
            model_score = float(model.score(index, investor, np.array([company]))[0])
            max_error = max(max_error, abs(sum(r.part for r in e.reasons) - model_score), abs(e.score - model_score))
            explanations.append((investor, e))

    chance = [inv for inv, e in explanations if e.by_chance]
    # Finding to measure: a startup missing a label has a shorter checklist, so its one match is not diluted
    ticks = {c: int(explainer._row(c).sum()) for c in {e.company for _, e in explanations}}
    missing_label = [e for _, e in explanations if ticks[e.company] < 2]
    labels = dataset.companies.set_index("permalink")[list(LABEL_COLUMNS)]
    return {"k": k, "dataset": dataset, "explanations": explanations, "picks": len(explanations),
            "test_investors": len(per_investor), "max_add_up_error": max_error,
            "chance_picks": len(chance), "missing_label_picks": len(missing_label),
            "board_missing_label_share": float(np.mean([explainer._row(c).sum() < 2 for c in board])), "investors_with_chance_picks": len(set(chance)),
            "company_labels": labels.to_dict("index")}


def format_report(output: dict) -> str:
    picks, chance = output["picks"], output["chance_picks"]
    two = sum(len(e.reasons) == 2 for _, e in output["explanations"])
    lines = [
        f"Phase 6: explanations for every top-{output['k']} pick of label similarity (2012 cutoff)",
        "",
        f"  picks explained:                {picks:,} ({output['test_investors']:,} investors)",
        f"  biggest add-up error:           {output['max_add_up_error']:.2e}",
        f"  matched on both labels:         {two:,} ({two / picks:.0%})",
        f"  matched on one label:           {picks - two - chance:,} ({(picks - two - chance) / picks:.0%})",
        f"  shown by chance (no match):     {chance:,} ({chance / picks:.1%}), "
        f"for {output['investors_with_chance_picks']:,} investors",
        f"  startup missing a label:        {output['missing_label_picks']:,} "
        f"({output['missing_label_picks'] / picks:.0%}; {output['board_missing_label_share']:.0%} of the board)",
        "",
        "Three examples:",
    ]
    for investor, e in output["explanations"][: 3 * output["k"] : output["k"]]:
        lines.append(f"  for {investor}: {e.company}, score {e.score:.3f}")
        lines.append(f"    {e.text}")
    return "\n".join(lines)


def main() -> None:
    print(format_report(run()))


if __name__ == "__main__":
    main()
