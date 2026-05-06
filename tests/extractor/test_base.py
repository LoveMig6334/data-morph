"""Unit tests for src/extractor/base.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.extractor.base import MetadataExtractor  # noqa: E402


class TestMetadataExtractorContract:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            MetadataExtractor()  # type: ignore[abstract]

    def test_schema_version_is_class_constant(self):
        assert MetadataExtractor.SCHEMA_VERSION == "0.1"

    def test_concrete_subclass_works(self):
        class Dummy(MetadataExtractor):
            def extract(self, file_path):
                return {"format": "dummy"}

            def supports(self, file_path):
                return True

        d = Dummy()
        assert d.supports(Path("x")) is True
        assert d.extract(Path("x")) == {"format": "dummy"}
