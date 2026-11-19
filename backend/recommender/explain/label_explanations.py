"""Explain a label similarity pick by taking its score apart.

A startup has at most two ticks: its category and its state. Cosine similarity is

    score = (profile . startup) / (|profile| x |startup|)

and the top line is just the sum over the startup's ticks. So the score splits
EXACTLY into one part per shared label:

    part(label) = profile[label] / (|profile| x |startup|)

Basketball: a box score. 18 points = 12 from threes + 6 from free throws, and
the parts must add up to the total. The explanation is the score itself, not
a story written afterwards.

A pick that shares no label with the investor scores 0 and only reaches the
top 10 through a random tie-break. It is labelled "shown by chance", never
given a made-up reason.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from recommender.models.label_similarity import LabelSimilarityRanker

MAX_EXAMPLES = 2


@dataclass(frozen=True)
class LabelReason:
    column: str          # "category_code" or "state_code"
    value: str           # e.g. "software", "CA"
    part: float          # this label's share of the score; all parts add up to the score
    past_deals: int      # how many of the investor's past companies carry this label
    examples: tuple      # up to two of those past companies, alphabetical


@dataclass(frozen=True)
class Explanation:
    company: str
    score: float
    reasons: tuple       # biggest part first; empty when nothing is shared
    by_chance: bool
    text: str


def _pretty(value: str) -> str:
    return str(value).replace("_", " ")


def _first_capital(text: str) -> str:
    """Capitalise the first letter only, so "CA" stays "CA" (str.capitalize would give "Ca")."""
    return text[:1].upper() + text[1:]


def _display_names(companies: pd.DataFrame | None) -> dict:
    """Company id -> name a person can read. Junk names (blank, or Excel's "#NAME?") are left out."""
    if companies is None or "name" not in companies.columns:
        return {}
    names = companies.set_index("permalink")["name"]
    good = names.notna() & ~names.astype(str).str.startswith("#NAME")
    return names[good].astype(str).to_dict()


class LabelExplainer:
    def __init__(self, model: LabelSimilarityRanker, train: pd.DataFrame, companies: pd.DataFrame | None = None):
        if model.profiles is None:
            raise RuntimeError("LabelExplainer needs a fitted LabelSimilarityRanker")
        self.model = model
        pairs = train[["investor_permalink", "company_permalink"]].drop_duplicates()
        self.past = {inv: sorted(group["company_permalink"]) for inv, group in pairs.groupby("investor_permalink")}
        self.label_names = model.vectors.labels   # e.g. "category_code=software"
        self.names = _display_names(companies)

    def _show(self, company: str) -> str:
        """Readable name, or the web address without "/company/" when there is no good name."""
        return self.names.get(company) or company.removeprefix("/company/")

    def _row(self, company: str) -> np.ndarray:
        index = self.model.vectors.index.get(company)
        width = self.model.vectors.matrix.shape[1]
        return np.zeros(width) if index is None else self.model.vectors.matrix[index].astype(float)

    def explain(self, investor: str, company: str) -> Explanation:
        profile = self.model.profiles.get(investor)
        if profile is None:
            raise KeyError(f"No training history for investor {investor!r}, so there is nothing to explain from")
        profile = profile.astype(float)
        row = self._row(company)
        lengths = np.linalg.norm(profile) * self.model.startup_length(row)

        reasons = []
        if lengths > 0:
            for j in np.flatnonzero(row):
                if profile[j] <= 0:
                    continue
                column, value = self.label_names[j].split("=", 1)
                sharing = [c for c in self.past[investor] if self._row(c)[j] > 0]
                reasons.append(LabelReason(column, value, float(profile[j] * row[j] / lengths),
                                           len(sharing), tuple(sharing[:MAX_EXAMPLES])))
        reasons.sort(key=lambda r: (-r.part, r.column))
        score = float(sum(r.part for r in reasons))
        by_chance = not reasons
        return Explanation(company, score, tuple(reasons), by_chance, self._text(row, reasons, score))

    def _text(self, row: np.ndarray, reasons: list, score: float) -> str:
        labels = dict(self.label_names[j].split("=", 1) for j in np.flatnonzero(row))
        category, state = labels.get("category_code"), labels.get("state_code")
        if category and state:
            head = f"{_first_capital(_pretty(category))} startup in {state}."
        elif category:
            head = f"{_first_capital(_pretty(category))} startup."
        elif state:
            head = f"Startup in {state}."
        else:
            head = "Startup with no labels."
        missing = ""
        if self.model.counts_missing_labels:
            gone = [c for c in self.model.columns if c not in labels]
            if gone and len(gone) < len(self.model.columns):
                names = " or ".join(c.removesuffix("_code") for c in gone)
                missing = f" No {names} on record, which counts as no match."
        if not reasons:
            return f"{head} No label match with your past deals. Shown by chance to fill the list."
        matches = " and ".join(
            f"your {_pretty(r.value)} deals ({r.past_deals}, e.g. {', '.join(self._show(c) for c in r.examples)})"
            for r in reasons
        )
        text = f"{head} Matches {matches}."
        if len(reasons) > 1:
            text += f" {_first_capital(_pretty(reasons[0].value))} counts for about {reasons[0].part / score:.0%} of the match."
        else:
            text += " That one label is the whole match."
        return text + missing
