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
    if row_count > 0:
        return None
    return MetadataWarning(
        code="EMPTY_FILE",
        severity="error",
        message="File has zero data rows.",
        context={"row_count": row_count},
    )


def check_missing_header(*, has_header: bool) -> MetadataWarning | None:
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
