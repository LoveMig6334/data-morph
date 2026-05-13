"""Unit tests for src/extractor/sampler.py."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.extractor.sampler import sample_csv  # noqa: E402


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


class TestSampleCSV:
    def test_default_split_on_10_rows(self, tmp_path: Path):
        path = tmp_path / "ten.csv"
        _write_csv(path, [["x"]] + [[str(i)] for i in range(10)])
        s = sample_csv(path, total_rows=10, encoding="utf-8")
        assert len(s["head"]) == 3
        assert len(s["middle"]) == 1
        assert len(s["tail"]) == 1
        # Tail should be the last row.
        assert s["tail"][0]["x"] == "9"
        # Head should be the first three rows.
        assert [r["x"] for r in s["head"]] == ["0", "1", "2"]

    def test_small_file_all_in_head(self, tmp_path: Path):
        path = tmp_path / "five.csv"
        _write_csv(path, [["x"]] + [[str(i)] for i in range(5)])
        s = sample_csv(path, total_rows=5, encoding="utf-8")
        assert len(s["head"]) == 5
        assert s["middle"] == []
        assert s["tail"] == []

    def test_boundary_exactly_at_sum(self, tmp_path: Path):
        # head_n + middle_n + tail_n = 5 by default → 5 rows triggers small-file rule.
        path = tmp_path / "five.csv"
        _write_csv(path, [["x"]] + [[str(i)] for i in range(5)])
        s = sample_csv(path, total_rows=5, encoding="utf-8")
        assert len(s["head"]) == 5
        assert s["middle"] == []
        assert s["tail"] == []

    def test_one_above_boundary(self, tmp_path: Path):
        path = tmp_path / "six.csv"
        _write_csv(path, [["x"]] + [[str(i)] for i in range(6)])
        s = sample_csv(path, total_rows=6, encoding="utf-8")
        assert len(s["head"]) == 3
        assert len(s["middle"]) == 1
        assert len(s["tail"]) == 1

    def test_empty_file_returns_empty_buckets(self, tmp_path: Path):
        path = tmp_path / "empty.csv"
        path.write_text("", encoding="utf-8")
        s = sample_csv(path, total_rows=0, encoding="utf-8")
        assert s == {"head": [], "middle": [], "tail": []}

    def test_custom_sizes(self, tmp_path: Path):
        path = tmp_path / "ten.csv"
        _write_csv(path, [["x"]] + [[str(i)] for i in range(10)])
        s = sample_csv(
            path,
            total_rows=10,
            encoding="utf-8",
            head_n=2,
            middle_n=2,
            tail_n=2,
        )
        assert len(s["head"]) == 2
        assert len(s["middle"]) == 2
        assert len(s["tail"]) == 2
