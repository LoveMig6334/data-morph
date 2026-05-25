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
