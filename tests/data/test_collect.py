from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.collect import PairResult, collect_case  # noqa: E402
from src.data.generators import uc3_txt_log_to_csv as uc3  # noqa: E402
from src.data.generators.base import write_case  # noqa: E402
from src.data.teacher_script import ScriptResult  # noqa: E402
from src.evaluation.runner import discover_cases  # noqa: E402

# A correct uc3 conversion script: [ts] LEVEL source: message -> CSV.
_GOOD_UC3_SCRIPT = '''
import sys, re, csv
src, dst = sys.argv[1], sys.argv[2]
pat = re.compile(r"^\\[(\\d{4}-\\d{2}-\\d{2}) (\\d{2}:\\d{2}:\\d{2})\\] (\\S+) (\\S+): (.*)$")
rows = []
for line in open(src, encoding="utf-8"):
    line = line.rstrip("\\n")
    if not line.strip():
        continue
    m = pat.match(line)
    if not m:
        continue
    d, t, level, source, msg = m.groups()
    rows.append((f"{d}T{t}", level, source.rstrip(":"), msg))
with open(dst, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, lineterminator="\\n")
    w.writerow(["timestamp", "level", "source", "message"])
    w.writerows(rows)
'''


def _one_case(tmp_path: Path):
    case = uc3.generate(seed=1, complexity="simple")
    write_case(case, tmp_path, "gen_000001")
    return discover_cases(tmp_path)[0]


def _fake_teacher(script: str, ok: bool = True):
    def fn(envelope, instruction, output_format, feedback=None):
        return ScriptResult("analysis", script if ok else "", "raw", 0 if ok else 1, "", {})
    return fn


class TestCollectCase:
    def test_good_script_accepted(self, tmp_path):
        case = _one_case(tmp_path)
        res = collect_case(case, teacher_fn=_fake_teacher(_GOOD_UC3_SCRIPT))
        assert isinstance(res, PairResult)
        assert res.accepted is True
        assert res.scores["format_validity"] == 1.0
        assert res.scores["content_accuracy"] >= 0.95
        assert res.retries == 0
        assert res.script.strip()

    def test_empty_script_rejected_after_retries(self, tmp_path):
        case = _one_case(tmp_path)
        res = collect_case(case, teacher_fn=_fake_teacher("", ok=False), max_retries=2)
        assert res.accepted is False
        assert res.retries == 2
        assert res.error_kind == "no_script"

    def test_broken_script_rejected_with_error_kind(self, tmp_path):
        case = _one_case(tmp_path)
        res = collect_case(case, teacher_fn=_fake_teacher("import sys\nraise ValueError('x')\n"), max_retries=1)
        assert res.accepted is False
        assert res.error_kind == "runtime"

    def test_envelope_has_no_file_path(self, tmp_path):
        # Captured envelope must not leak the local input path into training data.
        captured = {}

        def spy(envelope, instruction, output_format, feedback=None):
            captured.update(envelope)
            return ScriptResult("a", _GOOD_UC3_SCRIPT, "raw", 0, "", {})

        case = _one_case(tmp_path)
        collect_case(case, teacher_fn=spy)
        assert "file_path" not in captured
