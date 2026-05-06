"""Strategic head/middle/tail sampling for tabular files.

Reads only the rows it needs via pandas `nrows` and `skiprows`. Never
loads the full file into memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def sample_csv(
    file_path: Path,
    *,
    total_rows: int,
    encoding: str,
    head_n: int = 3,
    middle_n: int = 1,
    tail_n: int = 1,
) -> dict[str, list[dict[str, Any]]]:
    """Return head/middle/tail records as a dict of three lists.

    Small-file rule: if total_rows <= head_n + middle_n + tail_n, all rows
    go into head and middle/tail are empty.
    """
    if total_rows <= 0:
        return {"head": [], "middle": [], "tail": []}

    if total_rows <= head_n + middle_n + tail_n:
        head = pd.read_csv(file_path, nrows=total_rows, encoding=encoding, dtype=str)
        return {
            "head": head.to_dict("records"),
            "middle": [],
            "tail": [],
        }

    head = pd.read_csv(file_path, nrows=head_n, encoding=encoding, dtype=str)

    # middle: read middle_n rows starting near the file's midpoint
    middle_start = total_rows // 2
    middle = pd.read_csv(
        file_path,
        skiprows=range(1, middle_start + 1),
        nrows=middle_n,
        encoding=encoding,
        dtype=str,
    )

    # tail: skip everything but the last tail_n rows
    tail = pd.read_csv(
        file_path,
        skiprows=range(1, total_rows - tail_n + 1),
        nrows=tail_n,
        encoding=encoding,
        dtype=str,
    )

    return {
        "head": head.to_dict("records"),
        "middle": middle.to_dict("records"),
        "tail": tail.to_dict("records"),
    }
