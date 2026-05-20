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
