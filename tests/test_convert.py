"""Unit tests for the public convert_file API.

These exercise the full pipeline logic (envelope -> script -> sandbox -> validate
-> retry) without loading the model, by injecting a stub ``teacher_fn``. The real
model is covered by an opt-in smoke test at the bottom.
"""

from __future__ import annotations

import json

import pytest

from datamorph import ConversionResult, convert_file
from datamorph.data.teacher_script import ScriptResult
from datamorph.model import resolve_model

# A correct CSV -> JSON conversion script (reads argv[1], writes argv[2]).
GOOD_CSV_TO_JSON = """import sys, csv, json
rows = list(csv.DictReader(open(sys.argv[1])))
json.dump(rows, open(sys.argv[2], "w"))
"""

# A script that fails at runtime (never writes output).
BROKEN_SCRIPT = """import sys
raise ValueError("boom")
"""


def _script_result(script: str) -> ScriptResult:
    return ScriptResult("", script, script, 0, "", {})


def _stub(script: str):
    def teacher_fn(envelope, instruction, output_format, *, feedback=None):
        return _script_result(script)

    return teacher_fn


def _write_csv(path):
    path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    return path


def test_converts_and_writes_output(tmp_path):
    src = _write_csv(tmp_path / "in.csv")
    dst = tmp_path / "out.json"
    res = convert_file(src, dst, teacher_fn=_stub(GOOD_CSV_TO_JSON))

    assert isinstance(res, ConversionResult)
    assert res.accepted
    assert res.input_format == "csv" and res.output_format == "json"
    assert res.scores == {"format_validity": 1.0, "loadability": 1.0}
    assert res.retries == 0 and res.error is None
    assert res.output_path == dst
    assert json.loads(dst.read_text()) == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]


def test_formats_autodetected_from_extensions(tmp_path):
    src = _write_csv(tmp_path / "data.csv")
    res = convert_file(src, tmp_path / "data.json", teacher_fn=_stub(GOOD_CSV_TO_JSON))
    assert res.input_format == "csv" and res.output_format == "json"


def test_explicit_output_format_without_output_path(tmp_path):
    src = _write_csv(tmp_path / "in.csv")
    res = convert_file(src, input_format="csv", output_format="json",
                       teacher_fn=_stub(GOOD_CSV_TO_JSON))
    assert res.accepted
    assert res.output_path is None  # nothing written when no output_path given


def test_retries_on_failing_script_then_succeeds(tmp_path):
    src = _write_csv(tmp_path / "in.csv")
    calls = {"n": 0}

    def flaky(envelope, instruction, output_format, *, feedback=None):
        calls["n"] += 1
        # first attempt errors at runtime, then a working script
        return _script_result(BROKEN_SCRIPT if calls["n"] == 1 else GOOD_CSV_TO_JSON)

    res = convert_file(src, tmp_path / "out.json", teacher_fn=flaky, max_retries=3)
    assert res.accepted
    assert res.retries == 1
    assert calls["n"] == 2


def test_no_script_is_reported(tmp_path):
    src = _write_csv(tmp_path / "in.csv")
    res = convert_file(src, tmp_path / "out.json", teacher_fn=_stub(""), max_retries=1)
    assert not res.accepted
    assert res.error == "no_script"


def test_persistent_runtime_error_is_reported(tmp_path):
    src = _write_csv(tmp_path / "in.csv")
    res = convert_file(src, tmp_path / "out.json", teacher_fn=_stub(BROKEN_SCRIPT),
                       max_retries=1)
    assert not res.accepted
    assert res.error  # a sandbox error_kind (e.g. "runtime"), not None
    assert res.retries == 1


def test_unsupported_format_raises(tmp_path):
    bad = tmp_path / "in.xml"
    bad.write_text("<x/>", encoding="utf-8")
    with pytest.raises(ValueError):
        convert_file(bad, tmp_path / "out.json", teacher_fn=_stub(GOOD_CSV_TO_JSON))


def test_missing_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert_file(tmp_path / "nope.csv", tmp_path / "out.json",
                     teacher_fn=_stub(GOOD_CSV_TO_JSON))


def test_resolve_model_missing_path_raises():
    # A path-like string that doesn't exist (and isn't a repo id) errors clearly.
    with pytest.raises(FileNotFoundError):
        resolve_model("/definitely/not/a/real/model/path")


def test_resolve_model_returns_existing_local_path(tmp_path):
    assert resolve_model(str(tmp_path)) == str(tmp_path)


def test_resolve_model_downloads_explicit_repo_id(monkeypatch, tmp_path):
    called = {}

    def fake_download(repo_id, *args, **kwargs):
        called["repo"] = repo_id
        return str(tmp_path / "cache")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    out = resolve_model("Owner/some-model")
    assert called["repo"] == "Owner/some-model"
    assert out == str(tmp_path / "cache")


def test_resolve_model_falls_back_to_hub(monkeypatch, tmp_path):
    import datamorph.model as dm

    monkeypatch.delenv("GEMMA_MLX_MODEL", raising=False)
    monkeypatch.setattr(dm, "DEFAULT_MODEL_DIR", tmp_path / "no-local-model")
    called = {}

    def fake_download(repo_id, *args, **kwargs):
        called["repo"] = repo_id
        return str(tmp_path / "hub-cache")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    out = resolve_model()
    assert called["repo"] == dm.DEFAULT_HF_REPO
    assert out == str(tmp_path / "hub-cache")


@pytest.mark.performance
def test_real_model_on_test_set_case(tmp_path):
    """Opt-in smoke test of the real student through the public API.

    Runs a representative held-out case (with its prompt hint) end-to-end.
    Needs the model present locally; run with `pytest -m performance`.
    """
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    case = repo / "data" / "test_set" / "uc1_csv_to_json_nested" / "simple_01"
    meta = json.loads((case / "meta.json").read_text())
    res = convert_file(case / "input.csv", tmp_path / "out.json",
                       instruction=meta["prompt_hint"])  # default teacher -> real model
    assert res.accepted, f"conversion failed: {res.error} {res.scores}"
    assert res.scores["format_validity"] == 1.0 and res.scores["loadability"] == 1.0
