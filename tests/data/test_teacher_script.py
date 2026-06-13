from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestSkillFile:
    def test_skill_exists_and_states_contract(self):
        text = (PROJECT_ROOT / "skills" / "script_generation_teacher.md").read_text(encoding="utf-8")
        # The contract markers the teacher_script parser relies on:
        assert "<analysis>" in text and "</analysis>" in text
        assert "<script>" in text and "</script>" in text
        # The script I/O contract the sandbox relies on:
        assert "sys.argv[1]" in text and "sys.argv[2]" in text
        # Library constraint:
        assert "pandas" in text and "standard library" in text


from datamorph.data.teacher_script import (  # noqa: E402
    ScriptResult,
    build_script_prompt,
    parse_teacher_output,
)


class TestParseTeacherOutput:
    def test_extracts_analysis_and_script(self):
        raw = "<analysis>group by user</analysis>\n<script>import sys\nprint('x')\n</script>"
        analysis, script = parse_teacher_output(raw)
        assert analysis == "group by user"
        assert script == "import sys\nprint('x')"

    def test_strips_code_fence_inside_script(self):
        raw = "<analysis>a</analysis>\n<script>\n```python\nimport sys\n```\n</script>"
        _, script = parse_teacher_output(raw)
        assert script == "import sys"

    def test_missing_script_returns_empty(self):
        analysis, script = parse_teacher_output("<analysis>only analysis</analysis>")
        assert analysis == "only analysis"
        assert script == ""

    def test_handles_no_tags(self):
        analysis, script = parse_teacher_output("just some prose")
        assert analysis == "" and script == ""


class TestBuildScriptPrompt:
    def test_includes_envelope_instruction_and_io_contract(self):
        env = {"format": "csv", "schema": {"row_count": 3}}
        prompt = build_script_prompt(env, "Convert to nested JSON", "json")
        assert "script_generation_teacher.md" in prompt
        assert "Convert to nested JSON" in prompt
        assert "sys.argv[1]" in prompt and "sys.argv[2]" in prompt
        assert '"row_count": 3' in prompt  # envelope is embedded as JSON

    def test_feedback_is_appended_when_present(self):
        env = {"format": "csv"}
        prompt = build_script_prompt(env, "task", "json", feedback="output was empty")
        assert "output was empty" in prompt


class TestScriptResultOk:
    def test_ok_requires_returncode_zero_and_script(self):
        assert ScriptResult("a", "import sys", "raw", 0, "", {}).ok is True
        assert ScriptResult("a", "", "raw", 0, "", {}).ok is False
        assert ScriptResult("a", "import sys", "raw", 1, "err", {}).ok is False


import pytest  # noqa: E402


@pytest.mark.teacher
def test_live_opus_writes_runnable_script(tmp_path):
    """Opt-in: requires `claude` CLI + Opus access. Run with `-m teacher`."""
    from datamorph.data.collect import collect_case
    from datamorph.data.generators import uc3_txt_log_to_csv as uc3
    from datamorph.data.generators.base import write_case
    from datamorph.evaluation.runner import discover_cases

    case_obj = uc3.generate(seed=99, complexity="simple")
    write_case(case_obj, tmp_path, "gen_000001")
    case = discover_cases(tmp_path)[0]
    res = collect_case(case, max_retries=2)  # real Opus teacher (default teacher_fn)
    assert res.script, "teacher returned no script"
    assert res.accepted, f"pair not accepted: {res.error_kind} / {res.reason}"
