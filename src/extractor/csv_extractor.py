"""CSV metadata extractor — file inspection + schema inference + envelope.

CLI entry point at the bottom of the file.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

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


_BOOL_VALUES = {"true", "false", "True", "False", "0", "1", "yes", "no"}


def _is_null(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    if isinstance(v, str) and v == "":
        return True
    return False


def _try_int(v: str) -> bool:
    if not isinstance(v, str) or not v:
        return False
    # Reject leading zeros (e.g. "007" should stay string).
    s = v.lstrip("-")
    if len(s) > 1 and s.startswith("0"):
        return False
    try:
        int(v)
        return True
    except ValueError:
        return False


def _try_float(v: str) -> bool:
    if not isinstance(v, str) or not v:
        return False
    # Reject leading zeros (e.g. "007" should stay string).
    # Check the integer part: if it has leading zeros, reject.
    # Split on decimal point and check the left part.
    parts = v.lstrip("-").split(".")
    if parts:
        int_part = parts[0]
        if len(int_part) > 1 and int_part.startswith("0"):
            return False
    try:
        float(v)
        return True
    except ValueError:
        return False


def _try_bool(v: str) -> bool:
    return isinstance(v, str) and v in _BOOL_VALUES


def _try_date_column(values: list[str]) -> str | None:
    """Return 'date', 'datetime', or None for the column as a whole."""
    if not values:
        return None
    # Reject pure numbers with leading zeros (e.g. "007" should stay string).
    # But allow dates like "01/15/2026" which have non-digit chars.
    for v in values:
        if isinstance(v, str) and v:
            s = v.lstrip("-")
            # If the whole thing is digits and starts with 0, reject.
            if s.isdigit() and len(s) > 1 and s.startswith("0"):
                return None
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    nat_rate = parsed.isna().sum() / len(values)
    if nat_rate >= 0.05:
        return None
    has_time = any(
        (ts.hour, ts.minute, ts.second) != (0, 0, 0)
        for ts in parsed.dropna()
    )
    return "datetime" if has_time else "date"


def infer_column_dtype(values: list[str]) -> str:
    """Per spec §5.2: parser tagging then column-level resolution."""
    non_null = [v for v in values if not _is_null(v)]
    if not non_null:
        return "string"

    parsers = (
        ("integer", _try_int),
        ("float", _try_float),
        ("boolean", _try_bool),
    )
    accepts: dict[str, list[bool]] = {
        name: [parser(v) for v in non_null] for name, parser in parsers
    }

    # First parser (priority order) accepting every non-null value.
    for name, _ in parsers:
        if all(accepts[name]):
            return name

    # Date detection is column-level (NaT rate < 5 %).
    date_dtype = _try_date_column(non_null)
    if date_dtype is not None:
        return date_dtype

    # Some typed parser accepted at least one value but not all → mixed.
    if any(any(acc) for acc in accepts.values()):
        return "mixed"

    return "string"


def build_column_metadata(
    *,
    name: str,
    values: list[str],
    sample_values_per_column: int,
) -> dict[str, Any]:
    """Compute the per-column block of the schema field (§5.2)."""
    null_count = sum(1 for v in values if _is_null(v))
    non_null = [v for v in values if not _is_null(v)]
    unique_count = len(set(non_null))
    dtype = infer_column_dtype(values)

    samples: list[Any] = list(dict.fromkeys(non_null))[:sample_values_per_column]

    meta: dict[str, Any] = {
        "name": name,
        "dtype": dtype,
        "null_count": null_count,
        "unique_count": unique_count,
        "sample_values": _coerce_samples(samples, dtype),
    }

    if dtype == "integer":
        nums = [int(v) for v in non_null]
        meta["min"] = min(nums)
        meta["max"] = max(nums)
    elif dtype == "float":
        nums = [float(v) for v in non_null]
        meta["min"] = min(nums)
        meta["max"] = max(nums)
    elif dtype == "string":
        meta["max_length"] = max((len(v) for v in non_null), default=0)

    return meta


def _coerce_samples(samples: list[str], dtype: str) -> list[Any]:
    """Cast sample values to the column's inferred dtype for JSON output."""
    if dtype == "integer":
        return [int(v) for v in samples]
    if dtype == "float":
        return [float(v) for v in samples]
    if dtype == "boolean":
        return [v.lower() in ("true", "1", "yes") for v in samples]
    return list(samples)
