"""Unit tests for src/evaluation/metrics.py.

Each metric is tested with a passing case, a failing case, and edge cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable when running pytest from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import (  # noqa: E402
    content_accuracy,
    format_validity,
    loadability,
    schema_compliance,
    score_all,
)

# ---------------------------------------------------------------------------
# format_validity
# ---------------------------------------------------------------------------


class TestFormatValidity:
    def test_valid_json(self):
        assert format_validity('{"a": 1}', "json") == 1.0
        assert format_validity("[1, 2, 3]", "json") == 1.0

    def test_invalid_json(self):
        assert format_validity('{"a": }', "json") == 0.0
        assert format_validity("not json", "json") == 0.0
        assert format_validity("```json\n{}\n```", "json") == 0.0  # fences invalidate

    def test_valid_csv(self):
        assert format_validity("a,b\n1,2\n3,4\n", "csv") == 1.0

    def test_invalid_csv_ragged(self):
        # rows with different column counts -> invalid
        assert format_validity("a,b\n1,2,3\n", "csv") == 0.0

    def test_empty(self):
        assert format_validity("", "csv") == 0.0
        assert format_validity("", "json") == 0.0
        assert format_validity("", "txt") == 0.0

    def test_txt_always_passes_if_nonempty(self):
        assert format_validity("any text here", "txt") == 1.0


# ---------------------------------------------------------------------------
# schema_compliance
# ---------------------------------------------------------------------------


class TestSchemaCompliance:
    def test_json_same_structure(self):
        expected = '[{"user": {"name": "A"}, "orders": [{"id": 1}]}]'
        actual = '[{"user": {"name": "Zed"}, "orders": [{"id": 99}]}]'
        assert schema_compliance(actual, expected, "json") == 1.0

    def test_json_missing_key(self):
        expected = '{"a": 1, "b": 2}'
        actual = '{"a": 1}'
        assert schema_compliance(actual, expected, "json") == 0.0

    def test_json_extra_key_also_fails(self):
        expected = '{"a": 1}'
        actual = '{"a": 1, "extra": 2}'
        assert schema_compliance(actual, expected, "json") == 0.0

    def test_csv_header_match(self):
        expected = "name,age\nAlice,30\n"
        actual = "name,age\nZ,99\n"
        assert schema_compliance(actual, expected, "csv") == 1.0

    def test_csv_header_mismatch(self):
        expected = "name,age\nAlice,30\n"
        actual = "firstname,age\nAlice,30\n"
        assert schema_compliance(actual, expected, "csv") == 0.0

    def test_txt_always_passes(self):
        assert schema_compliance("anything", "anything else", "txt") == 1.0


# ---------------------------------------------------------------------------
# loadability
# ---------------------------------------------------------------------------


class TestLoadability:
    def test_csv_loads_in_pandas(self):
        assert loadability("a,b\n1,2\n3,4\n", "csv") == 1.0

    def test_malformed_csv_fails(self):
        # truly malformed — pandas will raise
        assert loadability("", "csv") == 0.0

    def test_json_list_loads(self):
        assert loadability('[{"a": 1}, {"a": 2}]', "json") == 1.0

    def test_json_nested_loads(self):
        assert loadability('{"users": [{"name": "A"}]}', "json") == 1.0

    def test_invalid_json_fails(self):
        assert loadability("{bad}", "json") == 0.0


# ---------------------------------------------------------------------------
# content_accuracy
# ---------------------------------------------------------------------------


class TestContentAccuracy:
    def test_json_perfect_match(self):
        e = '[{"name": "Alice", "age": 30}]'
        a = '[{"name": "Alice", "age": 30}]'
        assert content_accuracy(a, e, "json") == 1.0

    def test_json_partial_match(self):
        e = '[{"name": "Alice", "age": 30}]'
        a = '[{"name": "Alice", "age": 31}]'  # 1 of 2 leaves match
        assert content_accuracy(a, e, "json") == 0.5

    def test_json_numeric_coercion(self):
        # "9.99" (string) should equal 9.99 (float) numerically
        e = '{"price": 9.99}'
        a = '{"price": "9.99"}'
        assert content_accuracy(a, e, "json") == 1.0

    def test_csv_perfect_match(self):
        e = "name,age\nAlice,30\nBob,25\n"
        a = "name,age\nAlice,30\nBob,25\n"
        assert content_accuracy(a, e, "csv") == 1.0

    def test_csv_one_wrong_cell(self):
        e = "name,age\nAlice,30\nBob,25\n"
        a = "name,age\nAlice,30\nBob,99\n"  # 3 of 4 cells match
        assert content_accuracy(a, e, "csv") == 0.75

    def test_csv_header_mismatch_zeros_out(self):
        e = "name,age\nAlice,30\n"
        a = "firstname,age\nAlice,30\n"
        assert content_accuracy(a, e, "csv") == 0.0

    def test_txt_substring_match(self):
        a = "Sales Report\nNorth 12500\nTotal: 43300"
        required = ["Sales Report", "North", "12500", "43300"]
        assert content_accuracy(a, "", "txt", required) == 1.0

    def test_txt_partial_substring(self):
        a = "North 12500"
        required = ["North", "12500", "East", "15200"]
        assert content_accuracy(a, "", "txt", required) == 0.5

    def test_txt_case_insensitive(self):
        a = "TOTAL SALES: 43300"
        required = ["total sales", "43300"]
        assert content_accuracy(a, "", "txt", required) == 1.0


# ---------------------------------------------------------------------------
# score_all
# ---------------------------------------------------------------------------


class TestScoreAll:
    def test_returns_all_four_keys(self):
        scores = score_all('{"a": 1}', '{"a": 1}', "json")
        assert set(scores.keys()) == {
            "format_validity",
            "schema_compliance",
            "loadability",
            "content_accuracy",
        }
        assert all(s == 1.0 for s in scores.values())

    def test_all_zero_on_broken_output(self):
        scores = score_all("garbage", '{"a": 1}', "json")
        assert scores["format_validity"] == 0.0
        assert scores["schema_compliance"] == 0.0
        assert scores["loadability"] == 0.0
        assert scores["content_accuracy"] == 0.0
