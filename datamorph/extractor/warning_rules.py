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


def check_mixed_dtype_column(*, column: dict[str, Any]) -> MetadataWarning | None:
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


def check_likely_date_column(*, column: dict[str, Any]) -> MetadataWarning | None:
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


# ---------------------------------------------------------------------------
# JSON-specific warning rules
# ---------------------------------------------------------------------------

import re as _re  # noqa: E402  (intentionally aliased to avoid clashing with `re` elsewhere)

# Imported lazily inside type hints — avoids a runtime import cycle since
# json_walker.py does not need warning_rules.py.
from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:
    from .json_walker import PathStats


_ISO_DATE_RE = _re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?)?$")
_US_DATE_RE = _re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")


def check_optional_key(*, stats: "PathStats") -> MetadataWarning | None:
    """Fire `OPTIONAL_KEY` (warn) when a path has presence strictly between 0 and 1."""
    if stats.denominator <= 0:
        return None
    presence = stats.occurrence_count / stats.denominator
    if presence >= 1.0 or presence <= 0.0:
        return None
    return MetadataWarning(
        code="OPTIONAL_KEY",
        severity="warn",
        message=(
            f"Path '{stats.path}' is present in {stats.occurrence_count} of "
            f"{stats.denominator} records ({presence:.1%}). Conversion code "
            f"should treat it as optional and provide a default for missing values."
        ),
        context={
            "path": stats.path,
            "presence": round(presence, 4),
            "occurrence_count": stats.occurrence_count,
            "denominator": stats.denominator,
        },
    )


def check_mixed_type_path(*, stats: "PathStats") -> MetadataWarning | None:
    """Fire `MIXED_TYPE_PATH` (error) when a path has multiple non-null real dtypes."""
    real = set(stats.dtypes_seen) - {"null"}
    if len(real) <= 1:
        return None
    return MetadataWarning(
        code="MIXED_TYPE_PATH",
        severity="error",
        message=(
            f"Path '{stats.path}' has values of multiple types: "
            f"{sorted(real)}. Conversion code must explicitly coerce to a single type."
        ),
        context={"path": stats.path, "dtypes_seen": sorted(real)},
    )


def check_deeply_nested(*, max_depth: int, threshold: int) -> MetadataWarning | None:
    """Fire `DEEPLY_NESTED` (warn) when max_depth strictly exceeds threshold."""
    if max_depth <= threshold:
        return None
    return MetadataWarning(
        code="DEEPLY_NESTED",
        severity="warn",
        message=(
            f"JSON nesting reaches depth {max_depth}, exceeding the threshold "
            f"of {threshold}. Conversion code should use recursion or an explicit walker."
        ),
        context={"max_depth": max_depth, "threshold": threshold},
    )


def check_large_array(*, stats: "PathStats", threshold: int) -> MetadataWarning | None:
    """Fire `LARGE_ARRAY` (warn) when any observed array length strictly exceeds threshold."""
    if not stats.array_lengths_seen:
        return None
    max_len = max(stats.array_lengths_seen)
    if max_len <= threshold:
        return None
    return MetadataWarning(
        code="LARGE_ARRAY",
        severity="warn",
        message=(
            f"Array at '{stats.path}' has up to {max_len} elements (threshold "
            f"{threshold}). Generated script should stream this array, not load it whole."
        ),
        context={
            "path": stats.path,
            "max_length": max_len,
            "threshold": threshold,
        },
    )


def check_heterogeneous_array(
    *,
    stats: "PathStats",
    child_key_sets: list[frozenset[str]],
    mixed_element_types: bool = False,
) -> MetadataWarning | None:
    """Fire `HETEROGENEOUS_ARRAY` (warn) when array elements are non-uniform.

    Two triggers:
      1. `mixed_element_types` is True (array contains both objects and scalars).
      2. Jaccard similarity across `child_key_sets` is < 0.6.
    """
    if mixed_element_types:
        return MetadataWarning(
            code="HETEROGENEOUS_ARRAY",
            severity="warn",
            message=(
                f"Array at '{stats.path}' contains both object and non-object "
                f"elements. Conversion code must branch on element type."
            ),
            context={"path": stats.path, "reason": "mixed_element_types"},
        )
    if len(child_key_sets) < 2:
        return None
    # Pairwise minimum Jaccard
    min_jaccard = 1.0
    for i in range(len(child_key_sets)):
        for j in range(i + 1, len(child_key_sets)):
            a, b = child_key_sets[i], child_key_sets[j]
            union = a | b
            if not union:
                continue
            jaccard = len(a & b) / len(union)
            min_jaccard = min(min_jaccard, jaccard)
    if min_jaccard >= 0.6:
        return None
    return MetadataWarning(
        code="HETEROGENEOUS_ARRAY",
        severity="warn",
        message=(
            f"Array at '{stats.path}' has elements with non-uniform key sets "
            f"(min Jaccard {min_jaccard:.2f} < 0.60). Some keys exist only on some elements."
        ),
        context={
            "path": stats.path,
            "reason": "non_uniform_keys",
            "min_jaccard": round(min_jaccard, 3),
        },
    )


def check_likely_date_value(*, stats: "PathStats") -> MetadataWarning | None:
    """Fire `LIKELY_DATE_VALUE` (info) when ≥80% of sampled strings match a date pattern."""
    if "string" not in stats.dtypes_seen or len(stats.dtypes_seen - {"null"}) != 1:
        return None
    if not stats.sample_values:
        return None
    matches = sum(
        1
        for v in stats.sample_values
        if isinstance(v, str) and (_ISO_DATE_RE.match(v) or _US_DATE_RE.match(v))
    )
    ratio = matches / len(stats.sample_values)
    if ratio < 0.8:
        return None
    return MetadataWarning(
        code="LIKELY_DATE_VALUE",
        severity="info",
        message=(
            f"Path '{stats.path}' looks like a date column ({ratio:.0%} of "
            f"sampled values parse as ISO 8601 or MM/DD/YYYY). Conversion "
            f"script should parse + re-serialize consistently."
        ),
        context={"path": stats.path, "match_ratio": round(ratio, 2)},
    )


# ---------------------------------------------------------------------------
# TXT-specific warning rules
# ---------------------------------------------------------------------------


def check_no_pattern_detected(*, record_pattern: str) -> MetadataWarning | None:
    """Fire `NO_PATTERN_DETECTED` (warn) when no structured line pattern was found."""
    if record_pattern != "freeform":
        return None
    return MetadataWarning(
        code="NO_PATTERN_DETECTED",
        severity="warn",
        message=(
            "No consistent line structure (log/delimited/key-value/fixed-width) "
            "was detected. The conversion script must parse freeform text."
        ),
        context={"record_pattern": record_pattern},
    )


def check_mixed_line_structure(
    *, match_ratio: float, threshold: float = 0.8
) -> MetadataWarning | None:
    """Fire `MIXED_LINE_STRUCTURE` (warn) when only some lines match the dominant pattern."""
    if match_ratio <= 0.0 or match_ratio >= threshold:
        return None
    return MetadataWarning(
        code="MIXED_LINE_STRUCTURE",
        severity="warn",
        message=(
            f"Only {match_ratio:.0%} of lines match the dominant pattern "
            f"(threshold {threshold:.0%}). Some lines deviate — the script "
            f"should skip or special-case non-conforming lines."
        ),
        context={"match_ratio": round(match_ratio, 3), "threshold": threshold},
    )


def check_inconsistent_field_count(
    *, field_counts: list[int]
) -> MetadataWarning | None:
    """Fire `INCONSISTENT_FIELD_COUNT` (warn) when delimited lines split into varying widths."""
    if not field_counts or len(set(field_counts)) <= 1:
        return None
    return MetadataWarning(
        code="INCONSISTENT_FIELD_COUNT",
        severity="warn",
        message=(
            f"Delimited lines split into a varying number of fields "
            f"(observed widths: {sorted(set(field_counts))}). The script must "
            f"handle ragged rows."
        ),
        context={"observed_widths": sorted(set(field_counts))},
    )


def check_likely_timestamp_prefix(*, record_pattern: str) -> MetadataWarning | None:
    """Fire `LIKELY_TIMESTAMP_PREFIX` (info) when lines start with a timestamp."""
    if record_pattern != "log_line":
        return None
    return MetadataWarning(
        code="LIKELY_TIMESTAMP_PREFIX",
        severity="info",
        message=(
            "Lines begin with a timestamp prefix. The script should parse the "
            "leading timestamp and re-serialize it consistently (ISO 8601)."
        ),
        context={"record_pattern": record_pattern},
    )
