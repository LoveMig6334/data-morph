from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datamorph.data.sandbox import run_script  # noqa: E402

_GOOD = (
    "import sys\n"
    "src, dst = sys.argv[1], sys.argv[2]\n"
    "text = open(src).read().upper()\n"
    "open(dst, 'w').write(text)\n"
)
_SYNTAX = "def (:\n"  # invalid syntax
_RUNTIME = "import sys\nraise ValueError('boom')\n"
_NO_OUTPUT = "import sys\nprint('I never write the output file')\n"
_INFINITE = "import sys\nwhile True:\n    pass\n"


def _input(tmp_path: Path) -> Path:
    p = tmp_path / "in.txt"
    p.write_text("hello\n", encoding="utf-8")
    return p


class TestRunScript:
    def test_good_script_ok(self, tmp_path):
        res = run_script(_GOOD, _input(tmp_path), output_suffix=".txt")
        assert res.error_kind == "ok"
        assert res.ok is True
        assert res.output_text.strip() == "HELLO"

    def test_relative_input_path_resolved_against_caller_cwd(self, tmp_path, monkeypatch):
        # The sandbox runs the script with cwd=<tempdir>, so a RELATIVE input
        # path must be resolved against the caller's cwd, not the sandbox's,
        # or the script gets a FileNotFoundError. (Regression: the real CLI
        # passes data/raw/... relative paths; unit tests used absolute tmp_path.)
        monkeypatch.chdir(tmp_path)
        Path("in.txt").write_text("hello\n", encoding="utf-8")
        res = run_script(_GOOD, Path("in.txt"), output_suffix=".txt")
        assert res.error_kind == "ok", res.stderr
        assert res.output_text.strip() == "HELLO"

    def test_syntax_error(self, tmp_path):
        res = run_script(_SYNTAX, _input(tmp_path), output_suffix=".txt")
        assert res.error_kind == "syntax"
        assert res.ok is False

    def test_indentation_error_classified_as_syntax(self, tmp_path):
        # IndentationError is a SyntaxError subclass; its stderr label is
        # "IndentationError", not "SyntaxError" — must still classify as syntax.
        bad_indent = "import sys\nx = 1\n  y = 2\n"
        res = run_script(bad_indent, _input(tmp_path), output_suffix=".txt")
        assert res.error_kind == "syntax"
        assert "IndentationError" in res.stderr

    def test_runtime_error(self, tmp_path):
        res = run_script(_RUNTIME, _input(tmp_path), output_suffix=".txt")
        assert res.error_kind == "runtime"
        assert "boom" in res.stderr

    def test_no_output_file(self, tmp_path):
        res = run_script(_NO_OUTPUT, _input(tmp_path), output_suffix=".txt")
        assert res.error_kind == "empty_output"

    def test_timeout(self, tmp_path):
        res = run_script(_INFINITE, _input(tmp_path), output_suffix=".txt", timeout_sec=2.0)
        assert res.error_kind == "timeout"
