"""Tests for src.evaluation.output_cleanup.clean_model_output."""
from __future__ import annotations

import pytest

from src.evaluation.output_cleanup import clean_model_output


def test_pass_through_clean_csv():
    raw = "name,age\nAlice,30"
    cleaned, applied = clean_model_output(raw, "csv")
    assert cleaned == "name,age\nAlice,30"
    assert applied == []


def test_strip_json_code_fence():
    raw = '```json\n{"a":1}\n```'
    cleaned, applied = clean_model_output(raw, "json")
    assert cleaned == '{"a":1}'
    assert "strip_code_fence" in applied


def test_strip_csv_code_fence_drops_trailing_prose():
    raw = "```\nname\nA\n```\nThanks!"
    cleaned, applied = clean_model_output(raw, "csv")
    assert cleaned == "name\nA"
    assert "strip_code_fence" in applied


def test_strip_json_preamble():
    raw = "Here is the output:\n[{\"x\":1}]"
    cleaned, applied = clean_model_output(raw, "json")
    assert cleaned == '[{"x":1}]'
    assert "strip_preamble" in applied


def test_no_preamble_strip_for_txt():
    raw = "Report\n\nA  1\nB  2"
    cleaned, applied = clean_model_output(raw, "txt")
    assert cleaned == "Report\n\nA  1\nB  2"
    assert "strip_preamble" not in applied


def test_strip_trailing_prose_json():
    raw = '{"a":1}\nDone.'
    cleaned, applied = clean_model_output(raw, "json")
    assert cleaned == '{"a":1}'
    assert "strip_trailing_prose" in applied


def test_unclosed_fence_falls_through_to_preamble_strip():
    # Step 2 (fence) bails because no closing fence;
    # Step 3 (preamble) then drops the ```json line above the first {.
    raw = '```json\n{"a":1}'
    cleaned, applied = clean_model_output(raw, "json")
    assert cleaned == '{"a":1}'
    assert "strip_code_fence" not in applied
    assert "strip_preamble" in applied
