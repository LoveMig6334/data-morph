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
