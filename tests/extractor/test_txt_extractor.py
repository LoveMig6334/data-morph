from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.extractor.warning_rules import (  # noqa: E402
    check_no_pattern_detected,
    check_mixed_line_structure,
    check_inconsistent_field_count,
    check_likely_timestamp_prefix,
)


class TestNoPatternDetected:
    def test_fires_on_freeform(self):
        w = check_no_pattern_detected(record_pattern="freeform")
        assert w is not None
        assert w.code == "NO_PATTERN_DETECTED"
        assert w.severity == "warn"

    def test_silent_on_known_pattern(self):
        assert check_no_pattern_detected(record_pattern="log_line") is None


class TestMixedLineStructure:
    def test_fires_when_partial_match(self):
        w = check_mixed_line_structure(match_ratio=0.6, threshold=0.8)
        assert w is not None
        assert w.code == "MIXED_LINE_STRUCTURE"
        assert w.severity == "warn"

    def test_silent_at_full_match(self):
        assert check_mixed_line_structure(match_ratio=1.0, threshold=0.8) is None

    def test_silent_at_zero_match(self):
        # 0.0 means "no dominant pattern at all" — that's NO_PATTERN_DETECTED's job.
        assert check_mixed_line_structure(match_ratio=0.0, threshold=0.8) is None


class TestInconsistentFieldCount:
    def test_fires_on_varying_counts(self):
        w = check_inconsistent_field_count(field_counts=[3, 3, 4, 3])
        assert w is not None
        assert w.code == "INCONSISTENT_FIELD_COUNT"
        assert w.severity == "warn"

    def test_silent_on_uniform_counts(self):
        assert check_inconsistent_field_count(field_counts=[3, 3, 3]) is None

    def test_silent_on_empty(self):
        assert check_inconsistent_field_count(field_counts=[]) is None


class TestLikelyTimestampPrefix:
    def test_fires_on_log_line(self):
        w = check_likely_timestamp_prefix(record_pattern="log_line")
        assert w is not None
        assert w.code == "LIKELY_TIMESTAMP_PREFIX"
        assert w.severity == "info"

    def test_silent_otherwise(self):
        assert check_likely_timestamp_prefix(record_pattern="delimited") is None


from src.extractor.txt_extractor import infer_line_pattern  # noqa: E402


class TestInferLinePattern:
    def test_log_lines(self):
        lines = [
            "[2026-04-15 10:23:45] INFO app: Server started",
            "[2026-04-15 10:23:46] WARN db: Slow query",
        ]
        result = infer_line_pattern(lines)
        assert result["record_pattern"] == "log_line"
        assert result["match_ratio"] == 1.0

    def test_iso_log_lines_without_brackets(self):
        lines = [
            "2026-04-15T10:23:45 INFO started",
            "2026-04-15T10:23:46 WARN slow",
        ]
        assert infer_line_pattern(lines)["record_pattern"] == "log_line"

    def test_delimited(self):
        lines = ["a,b,c", "d,e,f", "g,h,i"]
        result = infer_line_pattern(lines)
        assert result["record_pattern"] == "delimited"
        assert result["delimiter"] == ","
        assert result["field_counts"] == [3, 3, 3]

    def test_key_value(self):
        lines = ["name: Alice", "age: 30", "city: NYC"]
        assert infer_line_pattern(lines)["record_pattern"] == "key_value"

    def test_freeform(self):
        lines = ["The quick brown fox", "jumped over the lazy dog"]
        assert infer_line_pattern(lines)["record_pattern"] == "freeform"

    def test_empty(self):
        result = infer_line_pattern([])
        assert result["record_pattern"] == "freeform"
        assert result["match_ratio"] == 0.0
