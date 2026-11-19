"""On main: tag a release only when the champion changed.

Prints the tag to create (or nothing) and writes the release notes.
A promotion is judged again here, on main, before it gets tagged.

    python -m recommender.gate.release --before before.json --after champion.json --notes notes.md
"""

import argparse
import sys
from pathlib import Path

from recommender.gate.check import check_promotion
from recommender.gate.cli import read_entry


def release_tag(before: dict | None, after: dict) -> str | None:
    return None if before == after else f"model-v{after['version']}"


def main(argv=None) -> int:
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--before", required=True)
    args.add_argument("--after", required=True)
    args.add_argument("--notes", required=True)
    a = args.parse_args(argv)
    before, after = read_entry(a.before), read_entry(a.after)
    tag = release_tag(before, after)
    if tag is None:
        print("Champion unchanged: no release.", file=sys.stderr)
        return 0
    result = check_promotion(before, after)
    if not result.passed:
        print(result.report, file=sys.stderr)
        print("The gate blocks this champion on main: no release.", file=sys.stderr)
        return 1
    Path(a.notes).write_text(result.report + "\n")
    print(tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
