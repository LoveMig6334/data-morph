"""Integration tests for src/extractor/csv_extractor.py."""

from __future__ import annotations

import sys
from pathlib import Path

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
