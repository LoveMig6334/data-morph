"""Unit tests for src/extractor/json_walker.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from src.extractor.json_walker import PathStats, walk  # noqa: E402


class TestPathStatsDataclass:
    def test_construction_with_all_fields(self):
        s = PathStats(
            path="[].id",
            dtypes_seen=frozenset({"integer"}),
            occurrence_count=5,
            denominator=5,
            sample_values=(1, 2, 3),
            max_depth_seen=1,
            array_lengths_seen=(),
            max_length=None,
            min_value=1,
            max_value=5,
            unique_count=3,
        )
        assert s.path == "[].id"
        assert s.dtypes_seen == frozenset({"integer"})
        assert s.occurrence_count == 5
        assert s.denominator == 5
        assert s.sample_values == (1, 2, 3)
        assert s.unique_count == 3

    def test_is_frozen(self):
        import dataclasses

        s = PathStats(
            path="x", dtypes_seen=frozenset({"string"}), occurrence_count=1,
            denominator=1, sample_values=("a",), max_depth_seen=0,
            array_lengths_seen=(), max_length=1, min_value=None,
            max_value=None, unique_count=1,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.path = "y"  # type: ignore[misc]


class TestWalkEmpty:
    def test_walk_returns_empty_for_scalar_root(self):
        """Scalar root is unsupported — walker returns empty list; extractor handles short-circuit."""
        assert walk("hello", sample_values_per_path=3) == []
        assert walk(42, sample_values_per_path=3) == []
        assert walk(None, sample_values_per_path=3) == []


class TestWalkArrayRootFlat:
    def test_path_notation_uses_brackets_for_array_root(self):
        result = walk(
            [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
            sample_values_per_path=3,
        )
        paths = {s.path for s in result}
        assert "[].id" in paths
        assert "[].name" in paths

    def test_occurrence_and_denominator_for_uniform_records(self):
        result = walk(
            [{"id": 1}, {"id": 2}, {"id": 3}],
            sample_values_per_path=3,
        )
        by_path = {s.path: s for s in result}
        assert by_path["[].id"].occurrence_count == 3
        assert by_path["[].id"].denominator == 3

    def test_sample_values_first_seen_wins(self):
        result = walk(
            [{"x": "a"}, {"x": "b"}, {"x": "c"}, {"x": "d"}],
            sample_values_per_path=2,
        )
        by_path = {s.path: s for s in result}
        assert by_path["[].x"].sample_values == ("a", "b")


class TestWalkObjectRoot:
    def test_object_root_paths_start_with_key(self):
        result = walk({"meta": {"version": 1}, "name": "x"}, sample_values_per_path=3)
        paths = {s.path for s in result}
        assert "name" in paths
        assert "meta.version" in paths

    def test_nested_object_path_uses_dot(self):
        result = walk(
            {"a": {"b": {"c": 42}}},
            sample_values_per_path=3,
        )
        paths = {s.path for s in result}
        assert "a.b.c" in paths

    def test_array_root_with_nested_object(self):
        result = walk(
            [{"user": {"name": "Alice", "age": 30}}],
            sample_values_per_path=3,
        )
        paths = {s.path for s in result}
        assert "[].user.name" in paths
        assert "[].user.age" in paths


class TestWalkNestedArrays:
    def test_nested_array_hop_uses_brackets(self):
        result = walk(
            [{"orders": [{"id": 1}, {"id": 2}]}, {"orders": [{"id": 3}]}],
            sample_values_per_path=5,
        )
        paths = {s.path for s in result}
        assert "[].orders" in paths
        assert "[].orders[].id" in paths

    def test_inner_array_denominator_is_total_inner_elements(self):
        result = walk(
            [{"orders": [{"id": 1}, {"id": 2}]}, {"orders": [{"id": 3}]}],
            sample_values_per_path=5,
        )
        by_path = {s.path: s for s in result}
        # 3 total order elements across both users
        assert by_path["[].orders[].id"].occurrence_count == 3
        assert by_path["[].orders[].id"].denominator == 3


class TestWalkContainerEntries:
    def test_object_container_path_has_object_dtype(self):
        result = walk({"meta": {"version": 1}}, sample_values_per_path=3)
        by_path = {s.path: s for s in result}
        assert "meta" in by_path
        assert "object" in by_path["meta"].dtypes_seen

    def test_array_container_has_array_lengths_seen(self):
        result = walk(
            [{"orders": [1, 2, 3]}, {"orders": [4, 5]}],
            sample_values_per_path=5,
        )
        by_path = {s.path: s for s in result}
        assert "[].orders" in by_path
        assert "array" in by_path["[].orders"].dtypes_seen
        assert sorted(by_path["[].orders"].array_lengths_seen) == [2, 3]


class TestWalkNumericExtras:
    def test_min_max_for_integer_path(self):
        result = walk([{"x": 1}, {"x": 5}, {"x": 3}], sample_values_per_path=3)
        by_path = {s.path: s for s in result}
        assert by_path["[].x"].min_value == 1
        assert by_path["[].x"].max_value == 5

    def test_min_max_for_float_path(self):
        result = walk([{"p": 1.5}, {"p": 0.5}, {"p": 3.5}], sample_values_per_path=3)
        by_path = {s.path: s for s in result}
        assert by_path["[].p"].min_value == 0.5
        assert by_path["[].p"].max_value == 3.5

    def test_max_length_for_string_path(self):
        result = walk(
            [{"n": "a"}, {"n": "abcd"}, {"n": "abc"}],
            sample_values_per_path=3,
        )
        by_path = {s.path: s for s in result}
        assert by_path["[].n"].max_length == 4

    def test_extras_unset_for_other_dtypes(self):
        result = walk([{"flag": True}, {"flag": False}], sample_values_per_path=3)
        by_path = {s.path: s for s in result}
        assert by_path["[].flag"].min_value is None
        assert by_path["[].flag"].max_value is None
        assert by_path["[].flag"].max_length is None


class TestWalkDtypeMixing:
    def test_null_alone_yields_only_null_dtype(self):
        result = walk([{"x": None}, {"x": None}], sample_values_per_path=3)
        by_path = {s.path: s for s in result}
        assert by_path["[].x"].dtypes_seen == frozenset({"null"})

    def test_null_mixed_with_string_is_not_promoted_to_mixed(self):
        # spec §5.5: "null-only contributions do not change the resolved dtype"
        # — we still record {"null", "string"} in dtypes_seen; the dtype
        # *resolution* (string vs mixed) happens in the extractor's
        # assembly step, not here.
        result = walk([{"x": None}, {"x": "a"}], sample_values_per_path=3)
        by_path = {s.path: s for s in result}
        assert by_path["[].x"].dtypes_seen == frozenset({"null", "string"})

    def test_two_real_dtypes_recorded(self):
        result = walk([{"x": 1}, {"x": "a"}], sample_values_per_path=3)
        by_path = {s.path: s for s in result}
        assert by_path["[].x"].dtypes_seen == frozenset({"integer", "string"})
