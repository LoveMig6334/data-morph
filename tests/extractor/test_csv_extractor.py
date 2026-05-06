"""Integration tests for src/extractor/csv_extractor.py."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.extractor.csv_extractor import (  # noqa: E402
    count_data_rows,
    detect_encoding,
    sniff_dialect,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestDetectEncoding:
    def test_utf8_succeeds_first(self, tmp_path: Path):
        p = tmp_path / "u.csv"
        p.write_text("a,b\n1,2\n", encoding="utf-8")
        enc, attempted = detect_encoding(p)
        # utf-8-sig is tried first; it succeeds on plain utf-8 too.
        # Implementation may report either "utf-8-sig" or "utf-8" — assert
        # the function picks one and reports its history correctly.
        assert enc in ("utf-8-sig", "utf-8")

    def test_bom_uses_utf8_sig(self, tmp_path: Path):
        p = tmp_path / "bom.csv"
        p.write_bytes(b"\xef\xbb\xbfa,b\n1,2\n")
        enc, attempted = detect_encoding(p)
        assert enc == "utf-8-sig"

    def test_latin1_fallback(self, tmp_path: Path):
        p = tmp_path / "latin.csv"
        # 0xe9 is é in latin-1 but invalid utf-8 start byte mid-sequence.
        p.write_bytes(b"name,value\nCaf\xe9,1\n")
        enc, attempted = detect_encoding(p)
        assert enc == "latin-1"
        assert "utf-8" in attempted
        assert "utf-8-sig" in attempted


class TestSniffDialect:
    def test_simple_users_has_header(self):
        d = sniff_dialect(FIXTURES / "simple_users.csv", encoding="utf-8")
        assert d["delimiter"] == ","
        assert d["has_header"] is True

    def test_headerless_detected(self):
        d = sniff_dialect(FIXTURES / "headerless.csv", encoding="utf-8")
        assert d["has_header"] is False


class TestCountDataRows:
    def test_with_header(self):
        # simple_users.csv has 5 data rows + 1 header row.
        n = count_data_rows(
            FIXTURES / "simple_users.csv", encoding="utf-8", has_header=True,
        )
        assert n == 5

    def test_without_header(self):
        # headerless.csv has 5 rows, no header.
        n = count_data_rows(
            FIXTURES / "headerless.csv", encoding="utf-8", has_header=False,
        )
        assert n == 5

    def test_empty_file(self):
        n = count_data_rows(
            FIXTURES / "empty_file.csv", encoding="utf-8", has_header=True,
        )
        assert n == 0


from src.extractor.csv_extractor import (  # noqa: E402
    build_column_metadata,
    infer_column_dtype,
)


class TestInferColumnDtype:
    def test_integer(self):
        assert infer_column_dtype(["1", "2", "3", "1001"]) == "integer"

    def test_integer_rejects_leading_zero(self):
        # "007" must stay string, not be reinterpreted as integer.
        assert infer_column_dtype(["007", "008"]) == "string"

    def test_float(self):
        assert infer_column_dtype(["1.5", "2.0", "3.14"]) == "float"

    def test_boolean_lower(self):
        assert infer_column_dtype(["true", "false", "true"]) == "boolean"

    def test_boolean_yes_no(self):
        assert infer_column_dtype(["yes", "no", "yes"]) == "boolean"

    def test_string_default(self):
        assert infer_column_dtype(["alice", "bob", "carol"]) == "string"

    def test_mixed(self):
        assert infer_column_dtype(["1", "two", "3"]) == "mixed"

    def test_date_iso(self):
        assert infer_column_dtype(["2026-01-15", "2026-02-20"]) == "date"

    def test_datetime_iso(self):
        assert infer_column_dtype(
            ["2026-01-15T10:30:00", "2026-02-20T14:45:00"]
        ) == "datetime"

    def test_date_us(self):
        assert infer_column_dtype(["01/15/2026", "02/20/2026"]) == "date"

    def test_empty_values_treated_as_null(self):
        assert infer_column_dtype(["1", "", "3"]) == "integer"

    def test_all_empty_is_string(self):
        assert infer_column_dtype(["", ""]) == "string"


class TestBuildColumnMetadata:
    def test_integer_column_includes_min_max(self):
        m = build_column_metadata(
            name="id", values=["1", "5", "3"], sample_values_per_column=3,
        )
        assert m["name"] == "id"
        assert m["dtype"] == "integer"
        assert m["min"] == 1
        assert m["max"] == 5
        assert m["null_count"] == 0
        assert m["unique_count"] == 3

    def test_float_column_includes_min_max(self):
        m = build_column_metadata(
            name="price", values=["9.99", "25.00"], sample_values_per_column=3,
        )
        assert m["min"] == 9.99
        assert m["max"] == 25.00

    def test_string_column_includes_max_length(self):
        m = build_column_metadata(
            name="name", values=["Al", "Bob", "Carol"],
            sample_values_per_column=3,
        )
        assert m["max_length"] == 5  # "Carol"
        assert "min" not in m
        assert "max" not in m

    def test_null_handling(self):
        m = build_column_metadata(
            name="x", values=["1", "", "3"], sample_values_per_column=3,
        )
        assert m["null_count"] == 1
        assert m["unique_count"] == 2

    def test_sample_values_capped(self):
        m = build_column_metadata(
            name="x", values=["a", "b", "c", "d", "e"],
            sample_values_per_column=3,
        )
        assert len(m["sample_values"]) == 3


from src.extractor.csv_extractor import CSVExtractor  # noqa: E402


def _codes(envelope: dict) -> list[str]:
    return [w["code"] for w in envelope["warnings"]]


class TestCSVExtractorEnvelope:
    def test_supports_csv(self):
        e = CSVExtractor()
        assert e.supports(Path("a.csv")) is True
        assert e.supports(Path("a.json")) is False

    def test_envelope_shape(self):
        e = CSVExtractor()
        env = e.extract(FIXTURES / "simple_users.csv")
        assert env["format"] == "csv"
        assert env["schema_version"] == "0.1"
        assert "file_size_bytes" in env
        assert env["encoding"] in ("utf-8", "utf-8-sig")
        assert "schema" in env and "samples" in env and "warnings" in env


class TestCSVExtractorWarnings:
    def test_simple_users_clean_baseline(self):
        env = CSVExtractor().extract(FIXTURES / "simple_users.csv")
        assert _codes(env) == [], (
            f"expected no warnings, got {env['warnings']}"
        )

    def test_repeating_entity_fixture_fires_expected_codes(self):
        env = CSVExtractor().extract(FIXTURES / "repeating_entity.csv")
        codes = _codes(env)
        assert "REPEATING_ENTITY" in codes
        assert "NUMERIC_COLUMN_QUOTE_RISK" in codes

    def test_numeric_columns_fixture(self):
        env = CSVExtractor().extract(FIXTURES / "numeric_columns.csv")
        codes = _codes(env)
        assert codes.count("NUMERIC_COLUMN_QUOTE_RISK") == 3

    def test_mixed_dtype_fixture(self):
        env = CSVExtractor().extract(FIXTURES / "mixed_dtype.csv")
        assert "MIXED_DTYPE_COLUMN" in _codes(env)

    def test_empty_file_fixture(self):
        env = CSVExtractor().extract(FIXTURES / "empty_file.csv")
        assert "EMPTY_FILE" in _codes(env)
        assert env["schema"]["row_count"] == 0

    def test_headerless_fixture_synthesizes_columns(self):
        env = CSVExtractor().extract(FIXTURES / "headerless.csv")
        assert "MISSING_HEADER" in _codes(env)
        names = [c["name"] for c in env["schema"]["columns"]]
        assert names == ["c0", "c1", "c2"]

    def test_duplicate_columns_fixture(self):
        env = CSVExtractor().extract(FIXTURES / "duplicate_columns.csv")
        assert "DUPLICATE_COLUMN_NAME" in _codes(env)
        names = [c["name"] for c in env["schema"]["columns"]]
        # pandas auto-renames the second "name" → "name.1".
        assert "name" in names and "name.1" in names


class TestCSVExtractorEncoding:
    def test_latin1_fallback_warning(self, tmp_path: Path):
        p = tmp_path / "latin.csv"
        p.write_bytes(b"name,city\nCaf\xe9,Paris\nNa\xefve,Lyon\n")
        env = CSVExtractor().extract(p)
        codes = _codes(env)
        assert "LATIN1_FALLBACK" in codes
        assert env["encoding"] == "latin-1"


class TestCSVExtractorSamples:
    def test_samples_present(self):
        env = CSVExtractor().extract(FIXTURES / "repeating_entity.csv")
        s = env["samples"]
        # 5 rows = head_n + middle_n + tail_n exactly → small-file rule.
        assert len(s["head"]) == 5
        assert s["middle"] == []
        assert s["tail"] == []


class TestCSVExtractorDelimiter:
    def test_tab_delimited_file_routes_through_pipeline(self, tmp_path: Path):
        """A TSV with duplicate column names must still trigger DUPLICATE_COLUMN_NAME."""
        p = tmp_path / "tabs.csv"  # .csv suffix but tab-delimited content
        p.write_text(
            "name\tname\tage\nAlice\tSmith\t30\nBob\tJones\t25\n",
            encoding="utf-8",
        )
        env = CSVExtractor().extract(p)
        codes = [w["code"] for w in env["warnings"]]
        # The dialect sniffer should detect tab, the raw-header reader
        # should split on tab, and DUPLICATE_COLUMN_NAME should fire.
        assert env["schema"]["delimiter"] == "\t", (
            f"expected tab delimiter, got {env['schema']['delimiter']!r}"
        )
        assert "DUPLICATE_COLUMN_NAME" in codes, (
            f"expected DUPLICATE_COLUMN_NAME, got {codes}"
        )
        # Pandas should produce 3 columns, not 1.
        assert len(env["schema"]["columns"]) == 3


@pytest.mark.performance
class TestCSVPerformance:
    def test_10k_rows_under_two_seconds(self):
        t0 = time.perf_counter()
        env = CSVExtractor().extract(FIXTURES / "large_file.csv")
        elapsed = time.perf_counter() - t0
        assert env["schema"]["row_count"] == 10_000
        assert elapsed <= 2.0, f"extraction took {elapsed:.2f}s (budget 2.0)"
