"""TXT metadata extractor — line-pattern inference + envelope.

CLI entry point at the bottom of the file. See the spec at
docs/superpowers/specs/2026-05-25-data-collection-pipeline-design.md.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from .base import MetadataExtractor
from .csv_extractor import detect_encoding
from .warning_rules import (
    MetadataWarning,
    check_empty_file,
    check_inconsistent_field_count,
    check_latin1_fallback,
    check_likely_timestamp_prefix,
    check_mixed_line_structure,
    check_no_pattern_detected,
)

PATTERN_THRESHOLD = 0.8
_DELIMITERS: tuple[str, ...] = (",", "\t", "|", ";")

# Leading timestamp: optional "[" then YYYY-MM-DD, space or T, HH:MM:SS, optional "]".
_LOG_PREFIX_RE = re.compile(r"^\[?\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\]?")
_KEY_VALUE_RE = re.compile(r"^[\w .-]{1,40}[:=]\s*\S")


def _dominant_delimiter(lines: list[str]) -> tuple[str | None, list[int]]:
    """Pick the delimiter that yields the most consistent (>=2) field counts."""
    best: tuple[str | None, list[int]] = (None, [])
    best_score = 0
    for delim in _DELIMITERS:
        counts = [len(ln.split(delim)) for ln in lines]
        multi = [c for c in counts if c >= 2]
        if not multi:
            continue
        # score = how many lines split into >=2 fields with the modal width
        modal = Counter(multi).most_common(1)[0][0]
        score = sum(1 for c in counts if c == modal)
        if score > best_score:
            best_score = score
            best = (delim, counts)
    return best


def infer_line_pattern(lines: list[str]) -> dict[str, Any]:
    """Classify the dominant line structure of non-blank lines.

    Returns a dict with at least `record_pattern` and `match_ratio`. For
    `log_line` adds `pattern_regex`; for `delimited` adds `delimiter` and
    `field_counts`.

    The dominant pattern is reported even when its match ratio is BELOW
    `PATTERN_THRESHOLD` (as long as it matched at least one line); that is what
    lets `check_mixed_line_structure` fire on a partial match. Only when no
    pattern matches any line is the result `freeform` with `match_ratio` 0.0.
    """
    if not lines:
        return {"record_pattern": "freeform", "match_ratio": 0.0}
    n = len(lines)

    log_ratio = sum(1 for ln in lines if _LOG_PREFIX_RE.match(ln)) / n

    delim, counts = _dominant_delimiter(lines)
    if delim is not None:
        multi = [c for c in counts if c >= 2]
        modal = Counter(multi).most_common(1)[0][0]
        delim_ratio = sum(1 for c in counts if c == modal) / n
    else:
        delim_ratio = 0.0

    kv_ratio = sum(1 for ln in lines if _KEY_VALUE_RE.match(ln)) / n

    # Priority on ties: log_line > delimited > key_value (max keeps the first).
    candidates = [("log_line", log_ratio), ("delimited", delim_ratio), ("key_value", kv_ratio)]
    best_pattern, best_ratio = max(candidates, key=lambda c: c[1])

    if best_ratio <= 0.0:
        return {"record_pattern": "freeform", "match_ratio": 0.0}

    result: dict[str, Any] = {
        "record_pattern": best_pattern,
        "match_ratio": round(best_ratio, 3),
    }
    if best_pattern == "log_line":
        result["pattern_regex"] = _LOG_PREFIX_RE.pattern
    elif best_pattern == "delimited":
        result["delimiter"] = delim
        result["field_counts"] = counts
    return result


def _read_nonblank_lines(file_path: Path, encoding: str) -> list[str]:
    """Return non-blank lines with trailing newlines stripped."""
    with file_path.open("r", encoding=encoding) as f:
        return [ln.rstrip("\n\r") for ln in f if ln.strip()]


def sample_lines(
    lines: list[str], *, head_n: int = 3, middle_n: int = 1, tail_n: int = 1
) -> dict[str, list[str]]:
    """Head/middle/tail sampling of lines with no overlap (mirrors sampler.sample_csv)."""
    n = len(lines)
    if n == 0:
        return {"head": [], "middle": [], "tail": []}
    if n <= head_n + middle_n + tail_n:
        return {"head": list(lines), "middle": [], "tail": []}
    head = lines[:head_n]
    tail = lines[n - tail_n:] if tail_n > 0 else []
    mid_start = (head_n + (n - tail_n) - middle_n) // 2
    middle = lines[mid_start : mid_start + middle_n] if middle_n > 0 else []
    return {"head": head, "middle": middle, "tail": tail}


class TXTExtractor(MetadataExtractor):
    """Stage 1c — turns a .txt/.log file into the shared metadata envelope."""

    def __init__(self, head_n: int = 3, middle_n: int = 1, tail_n: int = 1) -> None:
        self.head_n = head_n
        self.middle_n = middle_n
        self.tail_n = tail_n

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in (".txt", ".log")

    def extract(self, file_path: Path) -> dict[str, Any]:
        warnings: list[MetadataWarning] = []
        file_size = file_path.stat().st_size

        encoding, attempted = detect_encoding(file_path)
        _push(warnings, check_latin1_fallback(final_encoding=encoding, attempted=attempted))

        lines = _read_nonblank_lines(file_path, encoding)
        line_count = len(lines)
        _push(warnings, check_empty_file(row_count=line_count))

        pattern = infer_line_pattern(lines)
        _push(warnings, check_no_pattern_detected(record_pattern=pattern["record_pattern"]))
        _push(warnings, check_likely_timestamp_prefix(record_pattern=pattern["record_pattern"]))
        _push(
            warnings,
            check_mixed_line_structure(
                match_ratio=pattern.get("match_ratio", 0.0),
                threshold=PATTERN_THRESHOLD,
            ),
        )
        if pattern["record_pattern"] == "delimited":
            _push(
                warnings,
                check_inconsistent_field_count(field_counts=pattern.get("field_counts", [])),
            )

        schema: dict[str, Any] = {"line_count": line_count, **pattern}
        samples = sample_lines(
            lines, head_n=self.head_n, middle_n=self.middle_n, tail_n=self.tail_n
        )
        return {
            "format": "txt",
            "file_path": str(file_path),
            "file_size_bytes": file_size,
            "encoding": encoding,
            "schema_version": self.SCHEMA_VERSION,
            "schema": schema,
            "samples": samples,
            "warnings": [w.to_dict() for w in warnings],
        }


def _push(bucket: list[MetadataWarning], maybe: MetadataWarning | None) -> None:
    if maybe is not None:
        bucket.append(maybe)


def _main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="python -m datamorph.extractor.txt_extractor",
        description="Extract metadata envelope from a .txt/.log file.",
    )
    parser.add_argument("file", help="Path to a .txt or .log file")
    args = parser.parse_args()

    env = TXTExtractor().extract(Path(args.file))
    text = json.dumps(env, indent=2, default=str, ensure_ascii=False)
    print(text)
    print(f"# rough token estimate: ~{len(text) // 4} (chars / 4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
