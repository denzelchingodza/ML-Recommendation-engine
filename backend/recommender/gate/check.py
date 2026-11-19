"""Check a change to champion.json: is the promotion allowed?

champion.json looks like {"champion": "label_similarity", "version": 1}.

- No change: passes at once, the scoreboard does not run.
- No champion.json on the base branch: the first champion is recorded (must be
  version 1 and explainable). No contest, because it already earned the spot in Phases 2 to 6.
- A new champion: version must go up by exactly one, then the three rules decide.
"""

from dataclasses import dataclass

from recommender.gate.registry import EXPLAINABLE, KNOWN_MODELS
from recommender.gate.rules import decide, format_decision


class GateError(Exception):
    """champion.json asks for something the gate cannot allow."""


@dataclass(frozen=True)
class GateResult:
    passed: bool
    report: str


def _validate(entry: dict, where: str) -> None:
    if not isinstance(entry, dict) or "champion" not in entry:
        raise GateError(f"{where} champion.json must have a \"champion\" field")
    if not isinstance(entry.get("version"), int) or isinstance(entry.get("version"), bool):
        raise GateError(f"{where} champion.json must have a whole-number \"version\" field")
    if entry["champion"] not in KNOWN_MODELS:
        raise GateError(f"{where} champion.json names an unknown model: {entry['champion']!r}. "
                        f"Known models: {sorted(KNOWN_MODELS)}")


def check_promotion(base: dict | None, head: dict, runner=None, explainable=EXPLAINABLE) -> GateResult:
    _validate(head, "The new")
    if base is None:
        if head["version"] != 1:
            raise GateError("The first recorded champion must be version 1")
        if head["champion"] not in explainable:
            raise GateError(f"The first champion must be able to explain its picks; {head['champion']} cannot")
        return GateResult(True, f"## Quality gate\n\nFirst champion recorded: **{head['champion']}** (model v1). "
                                "It earned the spot on the scoreboards of Phases 2 to 6.")
    _validate(base, "The current")
    if head == base:
        return GateResult(True, f"## Quality gate\n\nNo promotion asked: the champion stays "
                                f"**{base['champion']}** (model v{base['version']}).")
    if head["champion"] == base["champion"]:
        raise GateError("The version changed but it is the same champion; only a new champion gets a new version")
    if head["version"] != base["version"] + 1:
        raise GateError(f"A promotion must be version {base['version'] + 1}, got {head['version']}")

    if runner is None:
        from recommender.evaluation.seasons import run as runner
    decision = decide(runner(), base["champion"], head["champion"], explainable)
    return GateResult(decision.passed, format_decision(decision))
