"""The draft board: which companies get ranked for each investor.

Board = every company that raised money on or after the cutoff.

Known catch (logged in docs/decisions.md): the board quietly tells us these
companies survived long enough to raise money after the cutoff. It does NOT
reveal who invested, which is the actual answer.
"""

import numpy as np
import pandas as pd


class DraftBoardError(Exception):
    """Raised when the board cannot be built honestly."""


def build_draft_board(investments: pd.DataFrame, cutoff: str) -> np.ndarray:
    """Sorted array of company ids with at least one dated deal on or after the cutoff."""
    dates = pd.to_datetime(investments["funded_at"], format="%Y-%m-%d", errors="coerce")
    after = investments.loc[dates >= pd.Timestamp(cutoff), "company_permalink"].dropna()
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
