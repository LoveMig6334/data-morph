from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.generators.base import GeneratedCase, make_faker, write_case  # noqa: E402


class TestMakeFaker:
    def test_seeded_faker_is_deterministic(self):
        a = make_faker(42).first_name()
        b = make_faker(42).first_name()
        assert a == b

    def test_different_seeds_differ(self):
        # Not guaranteed for every pair, but 7 vs 99999 should differ.
        assert make_faker(7).name() != make_faker(99999).name()


class TestWriteCase:
    def test_writes_three_files_in_test_set_layout(self, tmp_path):
        case = GeneratedCase(
            use_case="uc3_txt_log_to_csv",
            complexity="simple",
            input_format="txt",
            output_format="csv",
            input_text="[2026-04-15 10:00:00] INFO app: ok\n",
            expected_text="timestamp,level,source,message\n2026-04-15T10:00:00,INFO,app,ok\n",
            meta={"use_case": "uc3_txt_log_to_csv", "complexity": "simple"},
        )
        case_dir = write_case(case, tmp_path, "gen_000001")
        assert (case_dir / "input.txt").exists()
        assert (case_dir / "expected.csv").exists()
        meta = json.loads((case_dir / "meta.json").read_text())
        assert meta["use_case"] == "uc3_txt_log_to_csv"
        assert case_dir.parent.name == "uc3_txt_log_to_csv"
        assert case_dir.name == "gen_000001"
