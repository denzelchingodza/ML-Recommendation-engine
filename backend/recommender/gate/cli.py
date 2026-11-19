"""Command line for the gate, used by CI on every pull request.

    python -m recommender.gate --base base.json --head champion.json --report gate-report.md

Exit 0 = allowed, 1 = blocked by a rule, 2 = champion.json is invalid.
An empty or missing base file means there was no champion.json yet.
"""

import argparse
import json
import sys
from pathlib import Path

from recommender.gate.check import GateError, check_promotion


def read_entry(path: str) -> dict | None:
    p = Path(path)
    if not p.exists() or not p.read_text().strip():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as error:
        raise GateError(f"{path} is not valid JSON: {error}") from error


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--report", required=True)
    a = parser.parse_args(argv)
    try:
        head = read_entry(a.head)
        if head is None:
            raise GateError(f"{a.head} is missing or empty")
        result = check_promotion(read_entry(a.base), head)
    except GateError as error:
        message = f"## Quality gate\n\n**INVALID**: {error}"
        Path(a.report).write_text(message + "\n")
        print(message)
        return 2
    Path(a.report).write_text(result.report + "\n")
    print(result.report)
    return 0 if result.passed else 1
