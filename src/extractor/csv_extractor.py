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

    try:
        has_header = sniffer.has_header(sample)
    except csv.Error:
        has_header = True

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
