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


from .base import MetadataExtractor
from .sampler import sample_csv
from .warning_rules import (
    MetadataWarning,
    check_duplicate_column_name,
    check_empty_file,
    check_high_null_rate,
    check_inconsistent_quoting,
    check_latin1_fallback,
    check_likely_date_column,
    check_missing_header,
    check_mixed_dtype_column,
    check_numeric_column_quote_risk,
    check_repeating_entity,
)


class CSVExtractor(MetadataExtractor):
    """Concrete metadata extractor for CSV files (spec §4.2)."""

    def __init__(
        self,
        head_n: int = 3,
        middle_n: int = 1,
        tail_n: int = 1,
        sample_values_per_column: int = 3,
        max_rows_for_inference: int = 200,
    ) -> None:
        self.head_n = head_n
        self.middle_n = middle_n
        self.tail_n = tail_n
        self.sample_values_per_column = sample_values_per_column
        self.max_rows_for_inference = max_rows_for_inference

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".csv"

    def extract(self, file_path: Path) -> dict[str, Any]:
        warnings: list[MetadataWarning] = []
        file_size = file_path.stat().st_size

        encoding, attempted = detect_encoding(file_path)
        _push(warnings, check_latin1_fallback(
            final_encoding=encoding, attempted=attempted,
        ))

        # Empty file short-circuit.
        if file_size == 0:
            _push(warnings, check_empty_file(row_count=0))
            return self._envelope(
                file_path, file_size, encoding, schema={
                    "delimiter": ",", "quote_char": '"',
                    "has_header": False, "row_count": 0, "columns": [],
                },
                samples={"head": [], "middle": [], "tail": []},
                warnings=warnings,
            )

        dialect = sniff_dialect(file_path, encoding=encoding)
        _push(warnings, check_missing_header(has_header=dialect["has_header"]))
        _push(warnings, check_inconsistent_quoting(
            inconsistent=dialect["inconsistent_quoting"],
        ))

        row_count = count_data_rows(
            file_path, encoding=encoding, has_header=dialect["has_header"],
        )
        _push(warnings, check_empty_file(row_count=row_count))

        if row_count == 0:
            return self._envelope(
                file_path, file_size, encoding, schema={
                    **{k: dialect[k] for k in
                       ("delimiter", "quote_char", "has_header")},
                    "row_count": 0, "columns": [],
                },
                samples={"head": [], "middle": [], "tail": []},
                warnings=warnings,
            )

        # Read the raw header line first so we can detect duplicates BEFORE
        # pandas silently auto-renames them to 'name', 'name.1', etc.
        if dialect["has_header"]:
            with file_path.open("r", encoding=encoding, newline="") as f:
                raw_header = next(
                    csv.reader(f, delimiter=dialect["delimiter"]), []
                )
            _push(warnings, check_duplicate_column_name(raw_header=raw_header))

        # Schema inference on a capped sample.
        df = pd.read_csv(
            file_path,
            encoding=encoding,
            sep=dialect["delimiter"],
            quotechar=dialect["quote_char"],
            header=0 if dialect["has_header"] else None,
            nrows=self.max_rows_for_inference,
        )
        if not dialect["has_header"]:
            df.columns = [f"c{i}" for i in range(len(df.columns))]

        columns_meta: list[dict[str, Any]] = []
        for col_name in df.columns:
            raw_values = [
                "" if pd.isna(v) else str(v) for v in df[col_name].tolist()
            ]
            meta = build_column_metadata(
                name=str(col_name),
                values=raw_values,
                sample_values_per_column=self.sample_values_per_column,
            )
            columns_meta.append(meta)
            _push(warnings, check_repeating_entity(
                column=meta, row_count=row_count,
            ))
            _push(warnings, check_numeric_column_quote_risk(column=meta))
            _push(warnings, check_mixed_dtype_column(column=meta))
            _push(warnings, check_high_null_rate(
                column=meta, row_count=row_count,
            ))
            _push(warnings, check_likely_date_column(column=meta))

        samples = sample_csv(
            file_path,
            total_rows=row_count,
            encoding=encoding,
            head_n=self.head_n,
            middle_n=self.middle_n,
            tail_n=self.tail_n,
        )

        schema = {
            "delimiter": dialect["delimiter"],
            "quote_char": dialect["quote_char"],
            "has_header": dialect["has_header"],
            "row_count": row_count,
            "columns": columns_meta,
        }
        return self._envelope(
            file_path, file_size, encoding, schema, samples, warnings,
        )

    def _envelope(
        self,
        file_path: Path,
        file_size: int,
        encoding: str,
        schema: dict[str, Any],
        samples: dict[str, list[dict[str, Any]]],
        warnings: list[MetadataWarning],
    ) -> dict[str, Any]:
        return {
            "format": "csv",
            "file_path": str(file_path),
            "file_size_bytes": file_size,
            "encoding": encoding,
            "schema_version": self.SCHEMA_VERSION,
            "schema": schema,
            "samples": samples,
            "warnings": [w.to_dict() for w in warnings],
        }


def _push(
    bucket: list[MetadataWarning], maybe: MetadataWarning | None
) -> None:
    if maybe is not None:
        bucket.append(maybe)
