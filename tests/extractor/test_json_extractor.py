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


class TestEmptyFile:
    def test_byte_empty_file_short_circuits(self):
        ex = JSONExtractor()
        env = ex.extract(FIXTURES / "empty.json")
        codes = [w["code"] for w in env["warnings"]]
        assert codes == ["EMPTY_FILE"]
        assert env["schema"]["paths"] == []


class TestMalformedJson:
    def test_malformed_file_short_circuits(self):
        ex = JSONExtractor()
        env = ex.extract(FIXTURES / "malformed.json")
        codes = [w["code"] for w in env["warnings"]]
        assert codes == ["MALFORMED_JSON"]

    def test_non_utf8_bytes_treated_as_malformed(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_bytes(b"\xff\xfe\x00\x00not valid")
        ex = JSONExtractor()
        env = ex.extract(f)
        codes = [w["code"] for w in env["warnings"]]
        assert codes == ["MALFORMED_JSON"]


class TestRootShape:
    def test_record_array_classified(self):
        ex = JSONExtractor()
        env = ex.extract(FIXTURES / "simple_records.json")
        assert env["schema"]["root_shape"] == "record_array"
        assert env["schema"]["root_array_length"] == 5

    def test_object_root_classified(self):
        ex = JSONExtractor()
        env = ex.extract(FIXTURES / "nested_root.json")
        assert env["schema"]["root_shape"] == "object"
        assert "root_array_length" not in env["schema"]

    def test_scalar_root_short_circuits(self):
        ex = JSONExtractor()
        env = ex.extract(FIXTURES / "scalar_root.json")
        codes = [w["code"] for w in env["warnings"]]
        assert codes == ["ROOT_SHAPE_UNSUPPORTED"]
        assert env["schema"]["root_shape"] == "unsupported"

    def test_array_of_primitives_short_circuits(self, tmp_path):
        f = tmp_path / "prims.json"
        f.write_text("[1, 2, 3, 4, 5]", encoding="utf-8")
        ex = JSONExtractor()
        env = ex.extract(f)
        codes = [w["code"] for w in env["warnings"]]
        assert codes == ["ROOT_SHAPE_UNSUPPORTED"]

    def test_mostly_primitive_array_short_circuits(self, tmp_path):
        # <80% object elements → unsupported
        f = tmp_path / "mixed.json"
        f.write_text('[{"x":1}, 2, "a", 3, 4]', encoding="utf-8")
        ex = JSONExtractor()
        env = ex.extract(f)
        codes = [w["code"] for w in env["warnings"]]
        assert codes == ["ROOT_SHAPE_UNSUPPORTED"]
