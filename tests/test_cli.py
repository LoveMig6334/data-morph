"""Tests for the datamorph CLI.

Argument parsing and error handling run without the model; the end-to-end
conversion is covered by an opt-in performance test.
"""

from __future__ import annotations

import json

import pytest

from datamorph.cli import main


def _write_csv(path):
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    return path


def test_version_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "datamorph" in capsys.readouterr().out


def test_no_command_prints_help_and_returns_2(capsys):
    assert main([]) == 2
    assert "convert" in capsys.readouterr().err


def test_missing_input_returns_2(tmp_path, capsys):
    rc = main(["convert", str(tmp_path / "nope.csv"), str(tmp_path / "out.json")])
    assert rc == 2
    assert "error:" in capsys.readouterr().err


def test_unsupported_format_returns_2(tmp_path, capsys):
    bad = tmp_path / "in.xml"
    bad.write_text("<x/>", encoding="utf-8")
    rc = main(["convert", str(bad), str(tmp_path / "out.json")])
    assert rc == 2
    assert "error:" in capsys.readouterr().err


def test_output_without_format_returns_2(tmp_path, capsys):
    # no output path and no --output-format -> convert_file raises ValueError -> rc 2
    src = _write_csv(tmp_path / "in.csv")
    rc = main(["convert", str(src)])
    assert rc == 2


@pytest.mark.performance
def test_cli_converts_real_case(tmp_path, capsys):
    """Opt-in end-to-end CLI run against the real model. Run with `pytest -m performance`."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    case = repo / "data" / "test_set" / "uc1_csv_to_json_nested" / "simple_01"
    meta = json.loads((case / "meta.json").read_text())
    out = tmp_path / "out.json"
    rc = main(["convert", str(case / "input.csv"), str(out),
               "--instruction", meta["prompt_hint"]])
    assert rc == 0
    assert out.exists()
    json.loads(out.read_text())  # valid JSON
