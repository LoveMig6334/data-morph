from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.envelope import extract_envelope  # noqa: E402


class TestExtractEnvelope:
    def test_csv(self, tmp_path):
        p = tmp_path / "in.csv"
        p.write_text("a,b\n1,2\n", encoding="utf-8")
        env = extract_envelope(p, "csv")
        assert env["format"] == "csv"
        assert env["schema"]["row_count"] == 1

    def test_json(self, tmp_path):
        p = tmp_path / "in.json"
        p.write_text('[{"x": 1}]', encoding="utf-8")
        env = extract_envelope(p, "json")
        assert env["format"] == "json"

    def test_txt(self, tmp_path):
        p = tmp_path / "in.txt"
        p.write_text("[2026-04-15 10:00:00] INFO app: ok\n", encoding="utf-8")
        env = extract_envelope(p, "txt")
        assert env["format"] == "txt"

    def test_unknown_format_raises(self, tmp_path):
        import pytest

        with pytest.raises(KeyError):
            extract_envelope(tmp_path / "x.bin", "bin")
