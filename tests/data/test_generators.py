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


from src.evaluation.metrics import score_all  # noqa: E402


def _assert_oracle_self_consistent(case: GeneratedCase):
    """The expected output must be valid + self-consistent under the metrics."""
    required = case.meta.get("required_substrings")
    scores = score_all(
        case.expected_text, case.expected_text, case.output_format, required
    )
    assert scores["format_validity"] == 1.0
    assert scores["loadability"] == 1.0
    assert scores["content_accuracy"] == 1.0


class TestUC3:
    def test_deterministic(self):
        from src.data.generators import uc3_txt_log_to_csv as uc3

        a = uc3.generate(seed=1, complexity="simple")
        b = uc3.generate(seed=1, complexity="simple")
        assert a.input_text == b.input_text
        assert a.expected_text == b.expected_text

    def test_oracle_self_consistent(self):
        from src.data.generators import uc3_txt_log_to_csv as uc3

        for complexity in ("simple", "medium", "complex"):
            _assert_oracle_self_consistent(uc3.generate(seed=2, complexity=complexity))

    def test_shape(self):
        from src.data.generators import uc3_txt_log_to_csv as uc3

        case = uc3.generate(seed=3, complexity="simple")
        assert case.input_format == "txt"
        assert case.output_format == "csv"
        assert case.expected_text.splitlines()[0] == "timestamp,level,source,message"


class TestUC1:
    def test_deterministic(self):
        from src.data.generators import uc1_csv_to_json as uc1

        a = uc1.generate(seed=1, complexity="medium")
        b = uc1.generate(seed=1, complexity="medium")
        assert a.input_text == b.input_text and a.expected_text == b.expected_text

    def test_oracle_self_consistent(self):
        from src.data.generators import uc1_csv_to_json as uc1

        for c in ("simple", "medium", "complex"):
            _assert_oracle_self_consistent(uc1.generate(seed=2, complexity=c))

    def test_shape(self):
        from src.data.generators import uc1_csv_to_json as uc1

        case = uc1.generate(seed=3, complexity="simple")
        assert case.input_format == "csv" and case.output_format == "json"
        assert case.input_text.splitlines()[0] == (
            "user_name,user_email,order_id,order_item,order_price"
        )


class TestUC2:
    def test_deterministic(self):
        from src.data.generators import uc2_json_to_csv as uc2

        a = uc2.generate(seed=1, complexity="medium")
        b = uc2.generate(seed=1, complexity="medium")
        assert a.input_text == b.input_text and a.expected_text == b.expected_text

    def test_oracle_self_consistent(self):
        from src.data.generators import uc2_json_to_csv as uc2

        for c in ("simple", "medium", "complex"):
            _assert_oracle_self_consistent(uc2.generate(seed=2, complexity=c))

    def test_simple_header(self):
        from src.data.generators import uc2_json_to_csv as uc2

        case = uc2.generate(seed=3, complexity="simple")
        assert case.input_format == "json" and case.output_format == "csv"
        assert case.expected_text.splitlines()[0] == "name,address.city,address.zip"


class TestUC4:
    def test_deterministic(self):
        from src.data.generators import uc4_csv_to_txt_report as uc4

        a = uc4.generate(seed=1, complexity="medium")
        b = uc4.generate(seed=1, complexity="medium")
        assert a.input_text == b.input_text and a.expected_text == b.expected_text

    def test_oracle_self_consistent(self):
        from src.data.generators import uc4_csv_to_txt_report as uc4

        for c in ("simple", "medium", "complex"):
            _assert_oracle_self_consistent(uc4.generate(seed=2, complexity=c))

    def test_has_required_substrings(self):
        from src.data.generators import uc4_csv_to_txt_report as uc4

        case = uc4.generate(seed=3, complexity="simple")
        assert case.output_format == "txt"
        assert case.meta["content_accuracy_mode"] == "txt_substring"
        # Every required substring must actually appear in the report.
        for sub in case.meta["required_substrings"]:
            assert sub in case.expected_text
