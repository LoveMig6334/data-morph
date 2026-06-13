"""JSON metadata extractor — file inspection + path walker + envelope.

CLI entry point at the bottom of the file. See the spec at
docs/superpowers/specs/2026-05-20-json-metadata-extractor-design.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import MetadataExtractor
from .json_walker import PathStats, walk
from .warning_rules import (
    MetadataWarning,
    check_deeply_nested,
    check_heterogeneous_array,
    check_large_array,
    check_likely_date_value,
    check_mixed_type_path,
    check_optional_key,
)

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


def _resolve_dtype(dtypes_seen: frozenset[str]) -> str:
    """Resolve a path's dtype per spec §5.5.

    - All non-null contributions of one dtype → that dtype.
    - Multiple non-null dtypes → 'mixed'.
    - Only nulls → 'null'.
    """
    real = dtypes_seen - {"null"}
    if not real:
        return "null"
    if len(real) == 1:
        return next(iter(real))
    return "mixed"


def _sample_record_array(
    root: list[Any],
    head_n: int,
    middle_n: int,
    tail_n: int,
) -> dict[str, list[Any]]:
    """Return {head, middle, tail} per spec §5.3. No element is duplicated."""
    n = len(root)
    if n < head_n + middle_n + tail_n:
        return {"head": list(root), "middle": [], "tail": []}
    head = root[:head_n]
    tail = root[n - tail_n:] if tail_n > 0 else []
    if middle_n > 0:
        # Place middle in the center of the available space (between head and tail).
        available_start = head_n
        available_end = n - tail_n
        mid_start = (available_start + available_end - middle_n) // 2
        middle = root[mid_start : mid_start + middle_n]
    else:
        middle = []
    return {"head": head, "middle": middle, "tail": tail}


def _render_path(stats: PathStats) -> dict[str, Any]:
    """Render a PathStats into the envelope's per-path dict (spec §5.2)."""
    dtype = _resolve_dtype(stats.dtypes_seen)
    presence = (
        stats.occurrence_count / stats.denominator if stats.denominator > 0 else 0.0
    )
    entry: dict[str, Any] = {
        "path": stats.path,
        "dtype": dtype,
        "presence": round(presence, 4),
    }
    if dtype not in ("object", "array"):
        entry["sample_values"] = list(stats.sample_values)
        entry["unique_count"] = stats.unique_count
    if dtype in ("integer", "float") and stats.min_value is not None:
        entry["min"] = stats.min_value
        entry["max"] = stats.max_value
    if dtype == "string" and stats.max_length is not None:
        entry["max_length"] = stats.max_length
    return entry


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


def _split_path(path: str) -> list[str]:
    """Split a PathStats path string into a sequence of hops.

    Examples:
      ""               -> []
      "users"          -> ["users"]
      "[]"             -> ["[]"]
      "[].name"        -> ["[]", "name"]
      "users[].id"     -> ["users", "[]", "id"]
      "a.b.c"          -> ["a", "b", "c"]
    """
    if not path:
        return []
    out: list[str] = []
    buf = ""
    i = 0
    while i < len(path):
        ch = path[i]
        if ch == "[" and path[i : i + 2] == "[]":
            if buf:
                out.append(buf)
                buf = ""
            out.append("[]")
            i += 2
            if i < len(path) and path[i] == ".":
                i += 1
        elif ch == ".":
            if buf:
                out.append(buf)
                buf = ""
            i += 1
        else:
            buf += ch
            i += 1
    if buf:
        out.append(buf)
    return out


def _collect_array_element_signatures(
    root: Any, target_path: str
) -> tuple[list[frozenset[str]], bool]:
    """Return (child_key_sets, mixed_element_types_flag) for one array path.

    Walks `root` along `target_path` (using PathStats notation) and
    collects element key sets at every visit of the array. Used only
    by check_heterogeneous_array — too specialised to live in the walker.
    """
    parts = _split_path(target_path)

    key_sets: list[frozenset[str]] = []
    mixed_types = {"object": False, "non_object": False}

    def descend(node: Any, idx: int) -> None:
        if idx >= len(parts):
            # node is the array itself.
            if not isinstance(node, list):
                return
            for el in node:
                if isinstance(el, dict):
                    key_sets.append(frozenset(el.keys()))
                    mixed_types["object"] = True
                else:
                    mixed_types["non_object"] = True
            return
        part = parts[idx]
        if part == "[]":
            if isinstance(node, list):
                for el in node:
                    descend(el, idx + 1)
        else:
            if isinstance(node, dict) and part in node:
                descend(node[part], idx + 1)

    descend(root, 0)
    mixed = mixed_types["object"] and mixed_types["non_object"]
    return key_sets, mixed


def _collect_warnings(
    path_stats: list[PathStats],
    max_depth: int,
    max_depth_warn: int,
    max_array_len_warn: int,
    root: Any,
) -> list[MetadataWarning]:
    """Apply the six pure check_* rules over walker output + scalars."""
    warnings_: list[MetadataWarning] = []

    # Per-path rules.
    for s in path_stats:
        for fn in (
            check_optional_key,
            check_mixed_type_path,
            check_likely_date_value,
        ):
            w = fn(stats=s)
            if w is not None:
                warnings_.append(w)
        w = check_large_array(stats=s, threshold=max_array_len_warn)
        if w is not None:
            warnings_.append(w)

    # Whole-envelope rule: deeply-nested.
    w = check_deeply_nested(max_depth=max_depth, threshold=max_depth_warn)
    if w is not None:
        warnings_.append(w)

    # Heterogeneous-array rule needs per-array child-key-sets.
    for s in path_stats:
        if "array" not in s.dtypes_seen:
            continue
        key_sets, mixed_types = _collect_array_element_signatures(root, s.path)
        if not key_sets and not mixed_types:
            continue
        w = check_heterogeneous_array(
            stats=s,
            child_key_sets=key_sets,
            mixed_element_types=mixed_types,
        )
        if w is not None:
            warnings_.append(w)

    return warnings_


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

        # 7. Walk the root and render schema.paths.
        path_stats = walk(root, sample_values_per_path=self.sample_values_per_path)
        max_depth = max((s.max_depth_seen for s in path_stats), default=0)
        schema: dict[str, Any] = {
            "root_shape": root_shape,
            "max_depth": max_depth,
            "paths": [_render_path(s) for s in path_stats],
        }
        if root_shape == "record_array":
            schema["root_array_length"] = len(root)

        # 8. Sample records if root is a record_array.
        if root_shape == "record_array":
            samples = _sample_record_array(
                root, self.head_n, self.middle_n, self.tail_n
            )
        else:
            samples = {}

        # 9. Apply warning rules over walker output.
        warnings_list = _collect_warnings(
            path_stats=path_stats,
            max_depth=max_depth,
            max_depth_warn=self.max_depth_warn,
            max_array_len_warn=self.max_array_len_warn,
            root=root,
        )

        return {
            "format": "json",
            "file_path": str(file_path),
            "file_size_bytes": file_size,
            "encoding": "utf-8",
            "schema_version": MetadataExtractor.SCHEMA_VERSION,
            "schema": schema,
            "samples": samples,
            "warnings": [w.to_dict() for w in warnings_list],
        }


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m datamorph.extractor.json_extractor",
        description="Extract metadata envelope from a JSON file.",
    )
    parser.add_argument("file", help="Path to a .json file")
    args = parser.parse_args()

    env = JSONExtractor().extract(Path(args.file))
    text = json.dumps(env, indent=2, default=str, ensure_ascii=False)
    print(text)
    token_estimate = len(text) // 4
    print(f"# rough token estimate: ~{token_estimate} (chars / 4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
