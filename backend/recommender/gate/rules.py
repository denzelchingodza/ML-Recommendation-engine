"""The three gate rules, agreed with Denzel on 19 Nov 2026 BEFORE any challenger was judged.

1 Proven win: challenger minus champion, NDCG@10 overall, combined over all
  seasons with the cluster bootstrap: the 95% CI must sit entirely above zero.
  It must also not be a real loss on the fresh seasons (2009 to 2011).
2 Fair to new startups: in every season, the challenger's share of top-10 slots
  given to new startups is at most 5 points below the champion's.
  (Relative, because the current champion itself sits a little below the board share.)
3 Explainable: the challenger has an exact explainer.

Basketball: tryout rules are posted on the gym door before anyone plays.
"""

from dataclasses import dataclass

from recommender.evaluation.seasons import FRESH_SEASONS, combined_diff

FAIRNESS_MAX_DROP = 0.05
_EPS = 1e-12  # so "exactly 5 points below" passes despite floating point


@dataclass(frozen=True)
class RuleResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class Decision:
    champion: str
    challenger: str
    rules: tuple

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.rules)


def _proven_win(output: dict, champion: str, challenger: str) -> RuleResult:
    every = combined_diff(output, champion, challenger)
    fresh_cutoffs = tuple(c for c in FRESH_SEASONS if c in output["seasons"])
    fresh = combined_diff(output, champion, challenger, cutoffs=fresh_cutoffs) if fresh_cutoffs else every
    passed = every.low > 0 and fresh.high >= 0
    detail = (f"all seasons {every.mean:+.4f} (CI {every.low:+.4f} to {every.high:+.4f}); "
              f"fresh seasons {fresh.mean:+.4f} (CI {fresh.low:+.4f} to {fresh.high:+.4f})")
    if every.low <= 0:
        detail += ". The win is not proven: the CI does not sit entirely above zero"
    elif fresh.high < 0:
        detail += ". A real loss on the fresh seasons"
    return RuleResult("proven win", passed, detail)


def _fair(output: dict, champion: str, challenger: str) -> RuleResult:
    gaps = []
    for cutoff, season in output["seasons"].items():
        ch = season["results"][champion]["exposure"]["new_share_of_slots"]
        new = season["results"][challenger]["exposure"]["new_share_of_slots"]
        gaps.append((cutoff[:4], ch, new, new >= ch - FAIRNESS_MAX_DROP - _EPS))
    passed = all(ok for *_, ok in gaps)
    detail = "; ".join(f"{year}: {new:.0%} vs {ch:.0%}{'' if ok else ' (too low)'}" for year, ch, new, ok in gaps)
    return RuleResult("fair to new startups", passed, f"new-startup share, challenger vs champion: {detail}")


def _explainable(challenger: str, explainable) -> RuleResult:
    ok = challenger in explainable
    return RuleResult("explainable", ok, "has an exact explainer" if ok else
                      f"{challenger} has no exact explainer, so its picks could not be explained honestly")


def decide(output: dict, champion: str, challenger: str, explainable) -> Decision:
    return Decision(champion, challenger, (
        _proven_win(output, champion, challenger),
        _fair(output, champion, challenger),
        _explainable(challenger, explainable),
    ))


def format_decision(d: Decision) -> str:
    lines = [f"## Quality gate: {d.challenger} vs champion {d.champion}", "",
             f"**{'PASSED' if d.passed else 'BLOCKED'}**", ""]
    for r in d.rules:
        lines.append(f"- {'pass' if r.passed else 'FAIL'} · **{r.name}**: {r.detail}")
    return "\n".join(lines)
