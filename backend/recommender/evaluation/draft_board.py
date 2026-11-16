"""The draft board: which companies get ranked for each investor.

Board = every company that raised money on or after the cutoff
(and before `end`, when a season has a 12-month test window).

Known catch (logged in docs/decisions.md): the board quietly tells us these
companies survived long enough to raise money after the cutoff. It does NOT
reveal who invested, which is the actual answer.
"""

import numpy as np
import pandas as pd


class DraftBoardError(Exception):
    """Raised when the board cannot be built honestly."""


def build_draft_board(investments: pd.DataFrame, cutoff: str, end: str | None = None) -> np.ndarray:
    """Sorted array of company ids with at least one dated deal on or after the cutoff (and before end)."""
    dates = pd.to_datetime(investments["funded_at"], format="%Y-%m-%d", errors="coerce")
    inside = dates >= pd.Timestamp(cutoff)
    if end is not None:
        inside &= dates < pd.Timestamp(end)
    after = investments.loc[inside, "company_permalink"].dropna()
    board = np.array(sorted(set(after)))
    if len(board) == 0:
        raise DraftBoardError(f"No companies raised money on or after {cutoff}")
    return board


def check_board_covers_test(board: np.ndarray, test: pd.DataFrame) -> None:
    """Every hidden deal must be on the board, or it could never be found."""
    missing = set(test["company_permalink"]) - set(board)
    if missing:
        raise DraftBoardError(
            f"{len(missing)} hidden test companies are not on the draft board, e.g. {sorted(missing)[:3]}"
        )
