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
    """
    if not lines:
        return {"record_pattern": "freeform", "match_ratio": 0.0}
    n = len(lines)

    log_hits = sum(1 for ln in lines if _LOG_PREFIX_RE.match(ln))
    if log_hits / n >= PATTERN_THRESHOLD:
        return {
            "record_pattern": "log_line",
            "match_ratio": round(log_hits / n, 3),
            "pattern_regex": _LOG_PREFIX_RE.pattern,
        }

    delim, counts = _dominant_delimiter(lines)
    if delim is not None:
        multi = [c for c in counts if c >= 2]
        modal = Counter(multi).most_common(1)[0][0]
        match_ratio = sum(1 for c in counts if c == modal) / n
        if match_ratio >= PATTERN_THRESHOLD:
            return {
                "record_pattern": "delimited",
                "match_ratio": round(match_ratio, 3),
                "delimiter": delim,
                "field_counts": counts,
            }

    kv_hits = sum(1 for ln in lines if _KEY_VALUE_RE.match(ln))
    if kv_hits / n >= PATTERN_THRESHOLD:
        return {"record_pattern": "key_value", "match_ratio": round(kv_hits / n, 3)}

    return {"record_pattern": "freeform", "match_ratio": 0.0}
