"""Integration tests for src/extractor/json_extractor.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from src.extractor.json_extractor import JSONExtractor  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "json"


class TestJSONExtractorBasic:
    def test_supports_json_suffix(self):
        ex = JSONExtractor()
        assert ex.supports(Path("x.json")) is True
        assert ex.supports(Path("x.JSON")) is True
        assert ex.supports(Path("x.csv")) is False
        assert ex.supports(Path("x.txt")) is False


class TestFileTooLarge:
    def test_oversize_file_short_circuits(self, tmp_path):
        # 1-byte cap; any non-empty file is "too large".
        f = tmp_path / "big.json"
        f.write_text('[{"x":1}]', encoding="utf-8")
        ex = JSONExtractor(max_file_size_bytes=1)
        env = ex.extract(f)
        codes = [w["code"] for w in env["warnings"]]
        assert codes == ["FILE_TOO_LARGE"]
        assert env["format"] == "json"
        assert env["schema"]["paths"] == []
        assert env["samples"] == {}
