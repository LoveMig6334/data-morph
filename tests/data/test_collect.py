from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datamorph.data.collect import PairResult, collect_case, collect_corpus  # noqa: E402
from datamorph.data.generators import uc3_txt_log_to_csv as uc3  # noqa: E402
from datamorph.data.generators.base import write_case  # noqa: E402
from datamorph.data.teacher_script import ScriptResult  # noqa: E402
from datamorph.evaluation.runner import discover_cases  # noqa: E402

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

    def test_teacher_usage_captured_from_payload(self, tmp_path):
        # Opus token usage must be recorded (one-shot data — not recoverable later).
        usage = {"input_tokens": 1234, "output_tokens": 567}

        def teacher_with_usage(envelope, instruction, output_format, feedback=None):
            return ScriptResult("a", _GOOD_UC3_SCRIPT, "raw", 0, "", {"usage": usage})

        case = _one_case(tmp_path)
        res = collect_case(case, teacher_fn=teacher_with_usage)
        assert res.accepted is True
        assert res.teacher_usage == usage


class TestCollectCorpusResume:
    def _two_cases(self, tmp_path: Path) -> Path:
        raw = tmp_path / "raw"
        for i in (1, 2):
            write_case(uc3.generate(seed=i, complexity="simple"), raw, f"gen_00000{i}")
        return raw

    def test_resume_skips_existing_records_without_calling_teacher(self, tmp_path):
        raw = self._two_cases(tmp_path)
        interim = tmp_path / "interim"

        calls = {"n": 0}

        def counting_teacher(envelope, instruction, output_format, feedback=None):
            calls["n"] += 1
            return ScriptResult("analysis", _GOOD_UC3_SCRIPT, "raw", 0, "", {})

        # First pass: both cases attempted and accepted -> 2 teacher calls, 2 records.
        first = collect_corpus(raw, interim, teacher_fn=counting_teacher)
        assert first["n_accepted"] == 2
        assert first["n_skipped"] == 0
        assert calls["n"] == 2
        records = list(interim.glob("uc3_txt_log_to_csv__*.json"))
        assert len(records) == 2

        # Second pass with --resume: both already have records -> 0 teacher calls.
        calls["n"] = 0
        second = collect_corpus(raw, interim, teacher_fn=counting_teacher, resume=True)
        assert calls["n"] == 0
        assert second["n_skipped"] == 2
        assert second["n_attempted"] == 0
        assert second["n_accepted"] == 0
        assert second["n_records_total"] == 2

    def test_no_resume_reprocesses_everything(self, tmp_path):
        # Default (resume=False) must keep the original behaviour: every case attempted.
        raw = self._two_cases(tmp_path)
        interim = tmp_path / "interim"
        collect_corpus(raw, interim, teacher_fn=_fake_teacher(_GOOD_UC3_SCRIPT))

        calls = {"n": 0}

        def counting_teacher(envelope, instruction, output_format, feedback=None):
            calls["n"] += 1
            return ScriptResult("analysis", _GOOD_UC3_SCRIPT, "raw", 0, "", {})

        summary = collect_corpus(raw, interim, teacher_fn=counting_teacher)
        assert calls["n"] == 2  # re-attempted despite existing records
        assert summary["n_skipped"] == 0
        assert summary["accept_rate"] == 1.0
