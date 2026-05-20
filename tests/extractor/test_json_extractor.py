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


class TestSchemaPaths:
    def test_simple_records_paths(self):
        ex = JSONExtractor()
        env = ex.extract(FIXTURES / "simple_records.json")
        paths = {p["path"] for p in env["schema"]["paths"]}
        assert "[].id" in paths
        assert "[].name" in paths
        assert "[].email" in paths

    def test_path_entry_has_required_fields(self):
        ex = JSONExtractor()
        env = ex.extract(FIXTURES / "simple_records.json")
        id_entry = next(p for p in env["schema"]["paths"] if p["path"] == "[].id")
        assert id_entry["dtype"] == "integer"
        assert id_entry["presence"] == 1.0
        assert id_entry["min"] == 1
        assert id_entry["max"] == 5
        assert id_entry["sample_values"] == [1, 2, 3]
        assert "max_length" not in id_entry
        assert id_entry["unique_count"] == 3

    def test_string_path_has_max_length_no_min_max(self):
        ex = JSONExtractor()
        env = ex.extract(FIXTURES / "simple_records.json")
        name_entry = next(p for p in env["schema"]["paths"] if p["path"] == "[].name")
        assert name_entry["dtype"] == "string"
        assert "max_length" in name_entry
        assert "min" not in name_entry

    def test_max_depth_populated_for_nested_root(self):
        ex = JSONExtractor()
        env = ex.extract(FIXTURES / "nested_root.json")
        assert env["schema"]["max_depth"] >= 2

    def test_mixed_dtype_path_resolves_to_mixed(self, tmp_path):
        f = tmp_path / "m.json"
        f.write_text('[{"x":1},{"x":"a"}]', encoding="utf-8")
        ex = JSONExtractor()
        env = ex.extract(f)
        x = next(p for p in env["schema"]["paths"] if p["path"] == "[].x")
        assert x["dtype"] == "mixed"

    def test_null_only_dtype_resolves_to_null(self, tmp_path):
        f = tmp_path / "n.json"
        f.write_text('[{"x":null},{"x":null}]', encoding="utf-8")
        ex = JSONExtractor()
        env = ex.extract(f)
        x = next(p for p in env["schema"]["paths"] if p["path"] == "[].x")
        assert x["dtype"] == "null"

    def test_null_plus_string_resolves_to_string(self, tmp_path):
        f = tmp_path / "s.json"
        f.write_text('[{"x":null},{"x":"a"},{"x":"b"}]', encoding="utf-8")
        ex = JSONExtractor()
        env = ex.extract(f)
        x = next(p for p in env["schema"]["paths"] if p["path"] == "[].x")
        assert x["dtype"] == "string"


class TestSampling:
    def test_object_root_has_empty_samples(self):
        ex = JSONExtractor()
        env = ex.extract(FIXTURES / "nested_root.json")
        assert env["samples"] == {}

    def test_record_array_defaults_head_3_middle_1_tail_1(self):
        # simple_records.json has 5 records → 3 + 1 + 1 exactly.
        ex = JSONExtractor()
        env = ex.extract(FIXTURES / "simple_records.json")
        s = env["samples"]
        assert len(s["head"]) == 3
        assert len(s["middle"]) == 1
        assert len(s["tail"]) == 1
        # No overlap
        all_ids = [r["id"] for r in s["head"] + s["middle"] + s["tail"]]
        assert all_ids == sorted(set(all_ids))

    def test_small_file_rule_all_in_head(self, tmp_path):
        # 2 records, defaults are 3/1/1 → all 2 go in head, others empty.
        f = tmp_path / "tiny.json"
        f.write_text('[{"id":1},{"id":2}]', encoding="utf-8")
        ex = JSONExtractor()
        env = ex.extract(f)
        s = env["samples"]
        assert len(s["head"]) == 2
        assert s["middle"] == []
        assert s["tail"] == []

    def test_large_array_uses_middle_and_tail(self, tmp_path):
        records = [{"id": i} for i in range(100)]
        f = tmp_path / "many.json"
        f.write_text(json.dumps(records), encoding="utf-8")
        ex = JSONExtractor()
        env = ex.extract(f)
        s = env["samples"]
        assert [r["id"] for r in s["head"]] == [0, 1, 2]
        assert [r["id"] for r in s["tail"]] == [99]
        # middle is a single element from the middle of the array
        assert len(s["middle"]) == 1
        assert 40 <= s["middle"][0]["id"] <= 60


def _codes(env):
    return [w["code"] for w in env["warnings"]]


class TestWarningsBaseline:
    def test_simple_records_zero_warnings(self):
        ex = JSONExtractor()
        env = ex.extract(FIXTURES / "simple_records.json")
        assert _codes(env) == []

    def test_nested_root_zero_warnings(self):
        ex = JSONExtractor()
        env = ex.extract(FIXTURES / "nested_root.json")
        assert _codes(env) == []


class TestWarningOptionalKey:
    def test_fires_on_optional_keys_fixture(self):
        ex = JSONExtractor()
        env = ex.extract(FIXTURES / "optional_keys.json")
        codes = _codes(env)
        assert "OPTIONAL_KEY" in codes


class TestWarningMixedTypePath:
    def test_fires_on_mixed_type_fixture(self):
        ex = JSONExtractor()
        env = ex.extract(FIXTURES / "mixed_type_path.json")
        codes = _codes(env)
        assert "MIXED_TYPE_PATH" in codes


class TestWarningDeeplyNested:
    def test_fires_when_above_threshold(self):
        ex = JSONExtractor()
        env = ex.extract(FIXTURES / "deeply_nested.json")
        codes = _codes(env)
        assert "DEEPLY_NESTED" in codes

    def test_threshold_is_configurable(self):
        ex = JSONExtractor(max_depth_warn=100)
        env = ex.extract(FIXTURES / "deeply_nested.json")
        assert "DEEPLY_NESTED" not in _codes(env)


class TestWarningHeterogeneousArray:
    def test_fires_on_disjoint_key_sets(self):
        ex = JSONExtractor()
        env = ex.extract(FIXTURES / "heterogeneous_array.json")
        codes = _codes(env)
        assert "HETEROGENEOUS_ARRAY" in codes


class TestWarningLikelyDateValue:
    def test_fires_on_iso_date_strings(self):
        ex = JSONExtractor()
        env = ex.extract(FIXTURES / "dates.json")
        codes = _codes(env)
        assert "LIKELY_DATE_VALUE" in codes


class TestPerformance:
    @pytest.mark.performance
    def test_large_array_extracts_within_2s(self):
        import time

        ex = JSONExtractor()
        start = time.monotonic()
        env = ex.extract(FIXTURES / "large_array.json")
        elapsed = time.monotonic() - start
        assert env["schema"]["root_shape"] == "record_array"
        assert env["schema"]["root_array_length"] == 50_000
        # 2 s budget on M5 Max reference hardware.
        assert elapsed < 2.0, f"extraction took {elapsed:.2f}s (budget 2.0s)"
