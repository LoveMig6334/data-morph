"""CSV metadata extractor — file inspection + schema inference + envelope.

CLI entry point at the bottom of the file.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

ENCODING_LADDER: tuple[str, ...] = ("utf-8-sig", "utf-8", "latin-1")
SNIFF_SAMPLE_BYTES = 16 * 1024


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

    # Sniffer.has_header() heuristic is unreliable; use a hybrid approach.
    # Inspect the first row: if it looks like column names (contains underscores,
    # or all fields are shorter/simpler than the second row), assume header.
    has_header = True
    lines = sample.strip().split("\n")
    if len(lines) >= 2:
        try:
            reader = csv.reader(lines, delimiter=delimiter)
            first_row = next(reader, None)
            second_row = next(reader, None)
            if first_row and second_row:
                # Check if the first row contains underscores or looks like headers.
                first_has_underscore = any("_" in field for field in first_row)
                # Check if the first row has all purely numeric fields
                first_all_numeric_fields = all(
                    field.isdigit() for field in first_row if field
                )
                # If first row is all numeric (like "1,2,3,4"), it's likely not a header.
                if first_all_numeric_fields and second_row:
                    has_header = False
                elif first_has_underscore:
                    has_header = True
                else:
                    # Fallback: trust Sniffer's result or default to True.
                    try:
                        has_header = sniffer.has_header(sample)
                    except csv.Error:
                        has_header = True
            else:
                has_header = True
        except Exception:
            # If anything goes wrong, default to True.
            has_header = True
    else:
        has_header = True

    return {
        "delimiter": delimiter,
        "quote_char": quote_char,
        "has_header": has_header,
        # csv.Sniffer in the stdlib doesn't expose a consistent-quoting
        # check; this stays False unless we add deeper analysis later.
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
