"""Unit tests for the Gemma script-generation teacher (base-student baseline adapter).

The real MLX generation is mocked, so these run without the model or mlx_vlm.
"""

from __future__ import annotations

from dataclasses import dataclass

from datamorph.data.teacher_script import ScriptResult
from datamorph.models import gemma_script_teacher as gst


@dataclass
class _FakeGen:
    text: str
    model_id: str = "fake/gemma"
    n_prompt_tokens: int = 100
    n_generated_tokens: int = 50
    tokens_per_sec: float = 42.0
    elapsed_sec: float = 1.0
    truncated: bool = False


def _patch_generate(monkeypatch, text):
    monkeypatch.setattr(gst, "_SKILL_CACHE", {"text": "SKILL"})  # avoid disk read
    seen: list = []

    def fake(messages, max_tokens=4096):
        seen.append(messages)
        return _FakeGen(text=text)

    import datamorph.models.gemma_mlx as mlx
    monkeypatch.setattr(mlx, "generate", fake)
    return seen


def test_parses_analysis_and_script(monkeypatch):
    out = "<analysis>flat csv</analysis>\n<script>\nprint('hi')\n</script>"
    _patch_generate(monkeypatch, out)
    res = gst.call_gemma_script_teacher({"format": "csv"}, "Convert.", "json")
    assert isinstance(res, ScriptResult)
    assert res.analysis == "flat csv"
    assert res.script == "print('hi')"
    assert res.ok  # returncode 0 + non-empty script
    assert res.raw_payload["gemma_meta"]["n_generated_tokens"] == 50


def test_missing_script_is_not_ok(monkeypatch):
    _patch_generate(monkeypatch, "<analysis>no script here</analysis>")
    res = gst.call_gemma_script_teacher({"format": "csv"}, "Convert.", "json")
    assert res.script == ""
    assert not res.ok  # collect_case will record this as error_kind="no_script"


def test_prompt_includes_skill_envelope_and_contract(monkeypatch):
    seen = _patch_generate(monkeypatch, "<script>x=1</script>")
    gst.call_gemma_script_teacher({"format": "json"}, "Flatten it.", "csv")
    prompt = seen[0][0]["content"]
    assert "SKILL" in prompt
    assert '"format": "json"' in prompt
    assert "Flatten it." in prompt
    assert "sys.argv[1]" in prompt and "sys.argv[2]" in prompt


def test_generate_exception_returns_failed_result(monkeypatch):
    import datamorph.models.gemma_mlx as mlx
    monkeypatch.setattr(gst, "_SKILL_CACHE", {"text": "SKILL"})

    def boom(messages, max_tokens=4096):
        raise RuntimeError("mlx exploded")

    monkeypatch.setattr(mlx, "generate", boom)
    res = gst.call_gemma_script_teacher({"format": "csv"}, "Convert.", "json")
    assert not res.ok
    assert res.returncode == -1
    assert "mlx exploded" in res.stderr
