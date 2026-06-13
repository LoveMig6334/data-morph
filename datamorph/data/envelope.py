"""Dispatch a source file to the right Stage-1 extractor and return its envelope."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datamorph.extractor import CSVExtractor, JSONExtractor, MetadataExtractor, TXTExtractor

_EXTRACTORS: dict[str, type[MetadataExtractor]] = {
    "csv": CSVExtractor,
    "json": JSONExtractor,
    "txt": TXTExtractor,
}


def extract_envelope(input_path: Path, input_format: str) -> dict[str, Any]:
    """Return the metadata envelope for `input_path`, dispatching by `input_format`."""
    extractor_cls = _EXTRACTORS[input_format]  # KeyError on unknown format is intentional
    return extractor_cls().extract(input_path)
