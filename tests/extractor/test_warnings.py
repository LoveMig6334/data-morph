"""Unit tests for src/extractor/warning_rules.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.extractor.warning_rules import MetadataWarning  # noqa: E402


class TestMetadataWarning:
    def test_construction(self):
        w = MetadataWarning(
            code="EMPTY_FILE",
            severity="error",
            message="File is empty.",
            context={"row_count": 0},
        )
        assert w.code == "EMPTY_FILE"
        assert w.severity == "error"
        assert w.message == "File is empty."
        assert w.context == {"row_count": 0}

    def test_to_dict_round_trips(self):
        w = MetadataWarning(
            code="X",
            severity="warn",
            message="m",
            context={"k": 1},
        )
        d = w.to_dict()
        assert d == {
            "code": "X",
            "severity": "warn",
            "message": "m",
            "context": {"k": 1},
        }

    def test_is_frozen(self):
        import dataclasses

        w = MetadataWarning(code="X", severity="warn", message="m", context={})
        with pytest.raises(dataclasses.FrozenInstanceError):
            w.code = "Y"  # type: ignore[misc]


import pytest  # noqa: E402  (used in test_is_frozen)

from src.extractor.warning_rules import (  # noqa: E402
    check_empty_file,
    check_missing_header,
    check_duplicate_column_name,
    check_inconsistent_quoting,
    check_latin1_fallback,
)


class TestEmptyFile:
    def test_fires_on_zero_rows(self):
        w = check_empty_file(row_count=0)
        assert w is not None
        assert w.code == "EMPTY_FILE"
        assert w.severity == "error"

    def test_silent_on_nonzero(self):
        assert check_empty_file(row_count=5) is None


class TestMissingHeader:
    def test_fires_when_no_header(self):
        w = check_missing_header(has_header=False)
        assert w is not None
        assert w.code == "MISSING_HEADER"
        assert w.severity == "warn"

    def test_silent_when_header_present(self):
        assert check_missing_header(has_header=True) is None


class TestDuplicateColumnName:
    def test_fires_on_dupes(self):
        w = check_duplicate_column_name(raw_header=["name", "name", "email"])
        assert w is not None
        assert w.code == "DUPLICATE_COLUMN_NAME"
        assert w.context["duplicates"] == ["name"]

    def test_silent_on_unique(self):
        assert check_duplicate_column_name(raw_header=["a", "b", "c"]) is None


class TestInconsistentQuoting:
    def test_fires_when_flagged(self):
        w = check_inconsistent_quoting(inconsistent=True)
        assert w is not None
        assert w.code == "INCONSISTENT_QUOTING"

    def test_silent_when_consistent(self):
        assert check_inconsistent_quoting(inconsistent=False) is None


class TestLatin1Fallback:
    def test_fires_when_latin1_used(self):
        w = check_latin1_fallback(
            final_encoding="latin-1",
            attempted=["utf-8-sig", "utf-8"],
        )
        assert w is not None
        assert w.code == "LATIN1_FALLBACK"
        assert w.context["final_encoding"] == "latin-1"
        assert w.context["attempted_encodings"] == ["utf-8-sig", "utf-8"]

    def test_silent_when_utf8(self):
        assert check_latin1_fallback(final_encoding="utf-8", attempted=[]) is None


from src.extractor.warning_rules import (  # noqa: E402
    check_repeating_entity,
    check_numeric_column_quote_risk,
    check_mixed_dtype_column,
    check_high_null_rate,
    check_likely_date_column,
)


class TestRepeatingEntity:
    def test_fires_on_low_cardinality_string(self):
        col = {"name": "user_email", "dtype": "string", "unique_count": 47}
        w = check_repeating_entity(column=col, row_count=250)
        assert w is not None
        assert w.code == "REPEATING_ENTITY"
        assert w.context["rows_per_entity_avg"] == round(250 / 47, 2)

    def test_silent_on_high_cardinality(self):
        col = {"name": "id", "dtype": "string", "unique_count": 250}
        assert check_repeating_entity(column=col, row_count=250) is None

    def test_silent_on_numeric_columns(self):
        # Repeating-entity heuristic only applies to string columns.
        col = {"name": "score", "dtype": "integer", "unique_count": 5}
        assert check_repeating_entity(column=col, row_count=250) is None

    def test_silent_at_05_boundary(self):
        # ratio = 125 / 250 = 0.5 exactly → silent (rule fires only on < 0.5).
        col = {"name": "id", "dtype": "string", "unique_count": 125}
        assert check_repeating_entity(column=col, row_count=250) is None

    def test_silent_on_zero_unique(self):
        # All-null string column has unique_count == 0; must not crash.
        col = {"name": "comment", "dtype": "string", "unique_count": 0}
        assert check_repeating_entity(column=col, row_count=100) is None


class TestNumericColumnQuoteRisk:
    def test_fires_on_integer(self):
        col = {"name": "id", "dtype": "integer"}
        w = check_numeric_column_quote_risk(column=col)
        assert w is not None
        assert w.code == "NUMERIC_COLUMN_QUOTE_RISK"

    def test_fires_on_float(self):
        col = {"name": "price", "dtype": "float"}
        assert check_numeric_column_quote_risk(column=col) is not None

    def test_silent_on_string(self):
        col = {"name": "name", "dtype": "string"}
        assert check_numeric_column_quote_risk(column=col) is None


class TestMixedDtypeColumn:
    def test_fires_on_mixed(self):
        col = {"name": "value", "dtype": "mixed"}
        w = check_mixed_dtype_column(column=col)
        assert w is not None
        assert w.severity == "error"

    def test_silent_otherwise(self):
        col = {"name": "value", "dtype": "string"}
        assert check_mixed_dtype_column(column=col) is None


class TestHighNullRate:
    def test_fires_above_threshold(self):
        col = {"name": "comment", "null_count": 50}
        w = check_high_null_rate(column=col, row_count=250)
        assert w is not None
        assert w.code == "HIGH_NULL_RATE"

    def test_silent_at_threshold(self):
        col = {"name": "comment", "null_count": 25}  # 0.10 exactly
        assert check_high_null_rate(column=col, row_count=250) is None

    def test_silent_on_zero_rows(self):
        col = {"name": "comment", "null_count": 0}
        assert check_high_null_rate(column=col, row_count=0) is None


class TestLikelyDateColumn:
    def test_fires_on_date(self):
        col = {"name": "created_at", "dtype": "date"}
        assert check_likely_date_column(column=col) is not None

    def test_fires_on_datetime(self):
        col = {"name": "ts", "dtype": "datetime"}
        assert check_likely_date_column(column=col) is not None

    def test_silent_on_other(self):
        col = {"name": "name", "dtype": "string"}
        assert check_likely_date_column(column=col) is None


from src.extractor.warning_rules import (  # noqa: E402
    check_optional_key,
    check_mixed_type_path,
    check_deeply_nested,
    check_large_array,
    check_heterogeneous_array,
    check_likely_date_value,
)
from src.extractor.json_walker import PathStats  # noqa: E402


def _make_stats(**overrides) -> PathStats:
    """Build a PathStats with sensible defaults; override only what the test cares about."""
    defaults = dict(
        path="[].x",
        dtypes_seen=frozenset({"string"}),
        occurrence_count=10,
        denominator=10,
        sample_values=("a",),
        max_depth_seen=1,
        array_lengths_seen=(),
        max_length=1,
        min_value=None,
        max_value=None,
        unique_count=1,
    )
    defaults.update(overrides)
    return PathStats(**defaults)


class TestCheckOptionalKey:
    def test_full_presence_does_not_fire(self):
        s = _make_stats(occurrence_count=10, denominator=10)
        assert check_optional_key(stats=s) is None

    def test_partial_presence_fires(self):
        s = _make_stats(occurrence_count=9, denominator=10)
        w = check_optional_key(stats=s)
        assert w is not None
        assert w.code == "OPTIONAL_KEY"
        assert w.severity == "warn"
        assert w.context["presence"] == 0.9

    def test_zero_denominator_does_not_fire(self):
        s = _make_stats(occurrence_count=0, denominator=0)
        assert check_optional_key(stats=s) is None


class TestCheckMixedTypePath:
    def test_single_dtype_does_not_fire(self):
        s = _make_stats(dtypes_seen=frozenset({"string"}))
        assert check_mixed_type_path(stats=s) is None

    def test_string_plus_null_does_not_fire(self):
        s = _make_stats(dtypes_seen=frozenset({"string", "null"}))
        assert check_mixed_type_path(stats=s) is None

    def test_two_real_dtypes_fires(self):
        s = _make_stats(dtypes_seen=frozenset({"integer", "string"}))
        w = check_mixed_type_path(stats=s)
        assert w is not None
        assert w.code == "MIXED_TYPE_PATH"
        assert w.severity == "error"
        assert set(w.context["dtypes_seen"]) == {"integer", "string"}


class TestCheckDeeplyNested:
    def test_at_threshold_does_not_fire(self):
        assert check_deeply_nested(max_depth=6, threshold=6) is None

    def test_above_threshold_fires(self):
        w = check_deeply_nested(max_depth=7, threshold=6)
        assert w is not None
        assert w.code == "DEEPLY_NESTED"
        assert w.severity == "warn"
        assert w.context == {"max_depth": 7, "threshold": 6}


class TestCheckLargeArray:
    def test_at_threshold_does_not_fire(self):
        s = _make_stats(path="[]", dtypes_seen=frozenset({"array"}), array_lengths_seen=(10_000,))
        assert check_large_array(stats=s, threshold=10_000) is None

    def test_above_threshold_fires(self):
        s = _make_stats(path="[]", dtypes_seen=frozenset({"array"}), array_lengths_seen=(50_000,))
        w = check_large_array(stats=s, threshold=10_000)
        assert w is not None
        assert w.code == "LARGE_ARRAY"
        assert w.severity == "warn"
        assert w.context["max_length"] == 50_000


class TestCheckHeterogeneousArray:
    def test_uniform_keys_does_not_fire(self):
        s = _make_stats(path="[]", dtypes_seen=frozenset({"array"}))
        keys = [frozenset({"id", "name"}), frozenset({"id", "name"})]
        assert check_heterogeneous_array(stats=s, child_key_sets=keys) is None

    def test_disjoint_keys_fires(self):
        s = _make_stats(path="[]", dtypes_seen=frozenset({"array"}))
        keys = [frozenset({"a", "b"}), frozenset({"c", "d"})]
        w = check_heterogeneous_array(stats=s, child_key_sets=keys)
        assert w is not None
        assert w.code == "HETEROGENEOUS_ARRAY"
        assert w.severity == "warn"

    def test_mixed_object_and_scalar_fires(self):
        s = _make_stats(
            path="[]",
            dtypes_seen=frozenset({"array"}),
        )
        w = check_heterogeneous_array(
            stats=s, child_key_sets=[], mixed_element_types=True
        )
        assert w is not None
        assert w.code == "HETEROGENEOUS_ARRAY"


class TestCheckLikelyDateValue:
    def test_iso_8601_strings_fire(self):
        s = _make_stats(
            path="[].created_at",
            dtypes_seen=frozenset({"string"}),
            sample_values=("2026-05-20", "2026-05-21", "2026-05-22"),
        )
        w = check_likely_date_value(stats=s)
        assert w is not None
        assert w.code == "LIKELY_DATE_VALUE"
        assert w.severity == "info"

    def test_us_dates_fire(self):
        s = _make_stats(
            path="[].dob",
            dtypes_seen=frozenset({"string"}),
            sample_values=("05/20/2026", "06/01/1990"),
        )
        w = check_likely_date_value(stats=s)
        assert w is not None
        assert w.code == "LIKELY_DATE_VALUE"

    def test_non_date_strings_do_not_fire(self):
        s = _make_stats(
            path="[].name",
            dtypes_seen=frozenset({"string"}),
            sample_values=("Alice", "Bob"),
        )
        assert check_likely_date_value(stats=s) is None

    def test_non_string_paths_do_not_fire(self):
        s = _make_stats(
            path="[].n",
            dtypes_seen=frozenset({"integer"}),
            sample_values=(1, 2),
        )
        assert check_likely_date_value(stats=s) is None
