"""JSON metadata extractor — file inspection + path walker + envelope.

CLI entry point at the bottom of the file. See the spec at
docs/superpowers/specs/2026-05-20-json-metadata-extractor-design.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import MetadataExtractor
from .warning_rules import MetadataWarning

DEFAULT_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
DEFAULT_MAX_DEPTH_WARN = 6
DEFAULT_MAX_ARRAY_LEN_WARN = 10_000
OBJECT_ELEMENT_THRESHOLD = 0.8


def _classify_root(root: Any) -> str:
    """Return one of 'record_array', 'object', or 'unsupported' (spec §5.4)."""
    if isinstance(root, dict):
        return "object"
    if isinstance(root, list):
        if not root:
            return "unsupported"
        object_count = sum(1 for el in root if isinstance(el, dict))
        if object_count / len(root) >= OBJECT_ELEMENT_THRESHOLD:
            return "record_array"
        return "unsupported"
    return "unsupported"


def _minimal_envelope(
    file_path: Path,
    file_size: int,
    warnings_: list[MetadataWarning],
) -> dict[str, Any]:
    """Return a well-formed but empty envelope used by every short-circuit code."""
    return {
        "format": "json",
        "file_path": str(file_path),
        "file_size_bytes": file_size,
        "encoding": "utf-8",
        "schema_version": MetadataExtractor.SCHEMA_VERSION,
        "schema": {"root_shape": "unsupported", "paths": []},
        "samples": {},
        "warnings": [w.to_dict() for w in warnings_],
    }


class JSONExtractor(MetadataExtractor):
    """Stage 1b — turns a JSON file into the shared metadata envelope.

    See spec §4 for parameter semantics. All thresholds are configurable
    via the constructor; the defaults match spec §6.2.
    """

    def __init__(
        self,
        head_n: int = 3,
        middle_n: int = 1,
        tail_n: int = 1,
        sample_values_per_path: int = 3,
        max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE,
        max_depth_warn: int = DEFAULT_MAX_DEPTH_WARN,
        max_array_len_warn: int = DEFAULT_MAX_ARRAY_LEN_WARN,
    ) -> None:
        self.head_n = head_n
        self.middle_n = middle_n
        self.tail_n = tail_n
        self.sample_values_per_path = sample_values_per_path
        self.max_file_size_bytes = max_file_size_bytes
        self.max_depth_warn = max_depth_warn
        self.max_array_len_warn = max_array_len_warn

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".json"

    def extract(self, file_path: Path) -> dict[str, Any]:
        file_size = file_path.stat().st_size

        # 1. File-size guard (cheapest check, before any I/O beyond stat).
        if file_size > self.max_file_size_bytes:
            return _minimal_envelope(
                file_path,
                file_size,
                [
                    MetadataWarning(
                        code="FILE_TOO_LARGE",
                        severity="error",
                        message=(
                            f"File size {file_size} bytes exceeds cap "
                            f"{self.max_file_size_bytes} bytes."
                        ),
                        context={
                            "file_size_bytes": file_size,
                            "max_file_size_bytes": self.max_file_size_bytes,
                        },
                    )
                ],
            )

        # 2. Empty-file guard.
        if file_size == 0:
            return _minimal_envelope(
                file_path,
                file_size,
                [
                    MetadataWarning(
                        code="EMPTY_FILE",
                        severity="error",
                        message="File is byte-empty.",
                        context={"file_size_bytes": 0},
                    )
                ],
            )

        # 3. Decode (utf-8-sig handles BOM transparently).
        try:
            text = file_path.read_bytes().decode("utf-8-sig")
        except UnicodeDecodeError as e:
            return _minimal_envelope(
                file_path,
                file_size,
                [
                    MetadataWarning(
                        code="MALFORMED_JSON",
                        severity="error",
                        message=f"File is not valid UTF-8: {e}",
                        context={"error": str(e)},
                    )
                ],
            )

        # 4. Parse.
        try:
            root = json.loads(text)
        except json.JSONDecodeError as e:
            return _minimal_envelope(
                file_path,
                file_size,
                [
                    MetadataWarning(
                        code="MALFORMED_JSON",
                        severity="error",
                        message=f"JSON parse failed: {e}",
                        context={
                            "error": str(e),
                            "line": e.lineno,
                            "column": e.colno,
                        },
                    )
                ],
            )

        # 5. Empty-root guard (parsed [], {}, or null).
        if root in (None, [], {}):
            return _minimal_envelope(
                file_path,
                file_size,
                [
                    MetadataWarning(
                        code="EMPTY_FILE",
                        severity="error",
                        message="Parsed root is empty / null.",
                        context={"parsed_type": type(root).__name__},
                    )
                ],
            )

        # 6. Root-shape classification.
        root_shape = _classify_root(root)
        if root_shape == "unsupported":
            return _minimal_envelope(
                file_path,
                file_size,
                [
                    MetadataWarning(
                        code="ROOT_SHAPE_UNSUPPORTED",
                        severity="error",
                        message=(
                            f"JSON root is {type(root).__name__}; Phase 2 supports "
                            f"only object roots and record-array roots (≥80% object elements)."
                        ),
                        context={"observed_root_type": type(root).__name__},
                    )
                ],
            )

        # 7. Build the schema scaffold. Walker + warnings land in later tasks.
        schema: dict[str, Any] = {"root_shape": root_shape, "paths": []}
        if root_shape == "record_array":
            schema["root_array_length"] = len(root)
            schema["max_depth"] = 0  # populated in Task 12
        else:
            schema["max_depth"] = 0

        return {
            "format": "json",
            "file_path": str(file_path),
            "file_size_bytes": file_size,
            "encoding": "utf-8",
            "schema_version": MetadataExtractor.SCHEMA_VERSION,
            "schema": schema,
            "samples": {},
            "warnings": [],
        }
