"""CSV metadata extractor — file inspection + schema inference + envelope.

CLI entry point at the bottom of the file.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

ENCODING_LADDER: tuple[str, ...] = ("utf-8-sig", "utf-8", "latin-1")
SNIFF_SAMPLE_BYTES = 16 * 1024


def _is_numeric(value: str) -> bool:
    """Return True if value parses as int or float."""
    try:
        float(value)
        return True
    except ValueError:
        return False


def _body_has_numeric_column(sample: str, delimiter: str) -> bool:
    """Return True if the body (rows after row 1) has any all-numeric column.

    Used by sniff_dialect to disambiguate Sniffer's has_header verdict on
    all-string CSVs.
    """
    rows = list(csv.reader(sample.splitlines(), delimiter=delimiter))
    if len(rows) < 2:
        return False
    body = rows[1:]
    n_cols = len(rows[0])
    for col_idx in range(n_cols):
        col_values = [r[col_idx] for r in body if col_idx < len(r) and r[col_idx]]
        if not col_values:
            continue
        if all(_is_numeric(v) for v in col_values):
            return True
    return False


def detect_encoding(file_path: Path) -> tuple[str, list[str]]:
    """Return (chosen_encoding, list_of_encodings_attempted_before_success).

    Tries the ladder utf-8-sig → utf-8 → latin-1. latin-1 always succeeds.
    """
    attempted: list[str] = []
    for enc in ENCODING_LADDER:
        try:
            with file_path.open("r", encoding=enc) as f:
                f.read()
            return enc, attempted
        except UnicodeDecodeError:
            attempted.append(enc)
    # Should be unreachable: latin-1 cannot raise UnicodeDecodeError.
    raise RuntimeError("encoding ladder exhausted (this should never happen)")


def sniff_dialect(file_path: Path, *, encoding: str) -> dict[str, Any]:
    """Return delimiter / quote_char / has_header / inconsistent_quoting.

    Falls back to comma + double-quote + assumed-header if Sniffer fails
    (e.g. on very small or pathological files).
    """
    try:
        with file_path.open("r", encoding=encoding) as f:
            sample = f.read(SNIFF_SAMPLE_BYTES)
    except OSError:
        sample = ""

    if not sample:
        return {
            "delimiter": ",",
            "quote_char": '"',
            "has_header": False,
            "inconsistent_quoting": False,
        }

    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(sample)
        delimiter = dialect.delimiter
        quote_char = dialect.quotechar
    except csv.Error:
        delimiter = ","
        quote_char = '"'

    try:
        sniff_says_header = sniffer.has_header(sample)
    except csv.Error:
        sniff_says_header = True

    if sniff_says_header:
        has_header = True
    else:
        # Sniffer's has_header heuristic relies on a type difference between
        # row 1 and the body. On all-string CSVs it returns False even when
        # row 1 is a header. Apply one fallback: trust Sniffer only when the
        # body has at least one all-numeric column (strong type signal).
        # Otherwise default to has_header=True.
        has_header = not _body_has_numeric_column(sample, delimiter)

    return {
        "delimiter": delimiter,
        "quote_char": quote_char,
        "has_header": has_header,
        "inconsistent_quoting": False,
    }


def count_data_rows(
    file_path: Path, *, encoding: str, has_header: bool
) -> int:
    """Count non-blank data rows. Subtracts the header row if present."""
    try:
        with file_path.open("r", encoding=encoding) as f:
            n = sum(1 for line in f if line.strip())
    except OSError:
        return 0
    if has_header and n > 0:
        n -= 1
    return max(n, 0)
