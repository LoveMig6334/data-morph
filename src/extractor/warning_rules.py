"""Metadata warnings + pure detection rules for the extractor pipeline.

This module is pure: no file I/O, no network, no global state. Each
rule function takes the bare minimum data and returns a MetadataWarning
or None.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Severity = Literal["info", "warn", "error"]


@dataclass(frozen=True)
class MetadataWarning:
    """A single warning emitted by a metadata extractor.

    Attributes:
        code: Stable identifier (e.g. "REPEATING_ENTITY").
        severity: One of "info", "warn", "error".
        message: Human-readable message for the model and reviewer.
        context: Structured details for programmatic use.
    """

    code: str
    severity: Severity
    message: str
    context: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "context": dict(self.context),
        }


def check_empty_file(*, row_count: int) -> MetadataWarning | None:
    """Fire `EMPTY_FILE` (error) when the file has zero data rows."""
    if row_count > 0:
        return None
    return MetadataWarning(
        code="EMPTY_FILE",
        severity="error",
        message="File has zero data rows.",
        context={"row_count": row_count},
    )


def check_missing_header(*, has_header: bool) -> MetadataWarning | None:
    """Fire `MISSING_HEADER` (warn) when csv.Sniffer detects no header row."""
    if has_header:
        return None
    return MetadataWarning(
        code="MISSING_HEADER",
        severity="warn",
        message=(
            "csv.Sniffer reports no header row. Column names have been "
            "synthesized as c0, c1, ... — verify before relying on them."
        ),
        context={},
    )


def check_duplicate_column_name(*, raw_header: list[str]) -> MetadataWarning | None:
    """Fire `DUPLICATE_COLUMN_NAME` (warn) when header has repeated column names."""
    seen: set[str] = set()
    dupes: list[str] = []
    for name in raw_header:
        if name in seen and name not in dupes:
            dupes.append(name)
        seen.add(name)
    if not dupes:
        return None
    return MetadataWarning(
        code="DUPLICATE_COLUMN_NAME",
        severity="warn",
        message=(
            f"Header has duplicate column names: {dupes}. pandas auto-renames "
            f"duplicates with a numeric suffix (e.g. 'name', 'name.1')."
        ),
        context={"duplicates": dupes},
    )


def check_inconsistent_quoting(*, inconsistent: bool) -> MetadataWarning | None:
    """Fire `INCONSISTENT_QUOTING` (warn) when csv.Sniffer detects mixed quoting styles."""
    if not inconsistent:
        return None
    return MetadataWarning(
        code="INCONSISTENT_QUOTING",
        severity="warn",
        message=(
            "csv.Sniffer flagged inconsistent quoting in this file. The "
            "generated script should be robust to mixed quoting."
        ),
        context={},
    )


def check_latin1_fallback(
    *, final_encoding: str, attempted: list[str]
) -> MetadataWarning | None:
    """Fire `LATIN1_FALLBACK` (warn) when UTF-8 decoding fails and latin-1 is used as fallback."""
    if final_encoding != "latin-1":
        return None
    return MetadataWarning(
        code="LATIN1_FALLBACK",
        severity="warn",
        message=(
            "File could not be decoded as UTF-8. Fell back to latin-1, "
            "which may produce mojibake for non-Western characters."
        ),
        context={
            "attempted_encodings": list(attempted),
            "final_encoding": "latin-1",
        },
    )


def check_repeating_entity(
    *, column: dict[str, Any], row_count: int
) -> MetadataWarning | None:
    """Fire `REPEATING_ENTITY` (warn) when a string column has unique_count / row_count < 0.5."""
    if column.get("dtype") != "string":
        return None
    if row_count <= 0:
        return None
    unique_count = column["unique_count"]
    if unique_count == 0:
        return None
    ratio = unique_count / row_count
    if ratio >= 0.5:
        return None
    rows_per_entity_avg = round(row_count / unique_count, 2)
    return MetadataWarning(
        code="REPEATING_ENTITY",
        severity="warn",
        message=(
            f"Column '{column['name']}' has {unique_count} unique values "
            f"across {row_count} rows ({rows_per_entity_avg} rows per entity "
            f"avg). If converting to nested JSON, consider grouping rows by "
            f"this column before serializing."
        ),
        context={
            "column": column["name"],
            "unique_count": unique_count,
            "row_count": row_count,
            "rows_per_entity_avg": rows_per_entity_avg,
        },
    )


def check_numeric_column_quote_risk(
    *, column: dict[str, Any]
) -> MetadataWarning | None:
    """Fire `NUMERIC_COLUMN_QUOTE_RISK` (warn) when a numeric column may be incorrectly quoted."""
    dtype = column.get("dtype")
    if dtype not in ("integer", "float"):
        return None
    return MetadataWarning(
        code="NUMERIC_COLUMN_QUOTE_RISK",
        severity="warn",
        message=(
            f"Column '{column['name']}' is {dtype}. When serializing to "
            f"JSON, do NOT cast to string — preserve as numeric type."
        ),
        context={"column": column["name"], "dtype": dtype},
    )


def check_mixed_dtype_column(
    *, column: dict[str, Any]
) -> MetadataWarning | None:
    """Fire `MIXED_DTYPE_COLUMN` (error) when a column contains multiple data types."""
    if column.get("dtype") != "mixed":
        return None
    return MetadataWarning(
        code="MIXED_DTYPE_COLUMN",
        severity="error",
        message=(
            f"Column '{column['name']}' contains values of multiple dtypes. "
            f"The conversion script must perform explicit per-row casting."
        ),
        context={"column": column["name"]},
    )


def check_high_null_rate(
    *, column: dict[str, Any], row_count: int
) -> MetadataWarning | None:
    """Fire `HIGH_NULL_RATE` (warn) when a column has null_count / row_count > 0.1."""
    if row_count <= 0:
        return None
    null_count = column.get("null_count", 0)
    rate = null_count / row_count
    if rate <= 0.1:
        return None
    return MetadataWarning(
        code="HIGH_NULL_RATE",
        severity="warn",
        message=(
            f"Column '{column['name']}' has {null_count} nulls in "
            f"{row_count} rows ({rate:.1%}). Script must handle None/NaN."
        ),
        context={
            "column": column["name"],
            "null_count": null_count,
            "row_count": row_count,
            "null_rate": round(rate, 3),
        },
    )


def check_likely_date_column(
    *, column: dict[str, Any]
) -> MetadataWarning | None:
    """Fire `LIKELY_DATE_COLUMN` (info) when a column is detected as date or datetime type."""
    dtype = column.get("dtype")
    if dtype not in ("date", "datetime"):
        return None
    return MetadataWarning(
        code="LIKELY_DATE_COLUMN",
        severity="info",
        message=(
            f"Column '{column['name']}' is detected as {dtype}. Script "
            f"should parse and re-serialize using a consistent format "
            f"(ISO 8601 recommended)."
        ),
        context={"column": column["name"], "dtype": dtype},
    )
