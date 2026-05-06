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
