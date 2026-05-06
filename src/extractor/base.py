"""Abstract base class for format-specific metadata extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar


class MetadataExtractor(ABC):
    """Every format-specific extractor implements this contract.

    Subclasses produce a metadata dict in the shared envelope schema
    (see docs/superpowers/specs/2026-05-06-csv-metadata-extractor-design.md
    section 5.1).
    """

    SCHEMA_VERSION: ClassVar[str] = "0.1"

    @abstractmethod
    def extract(self, file_path: Path) -> dict[str, Any]:
        """Return a metadata dict in the shared envelope schema."""

    @abstractmethod
    def supports(self, file_path: Path) -> bool:
        """Return True iff this extractor can handle the given file."""
