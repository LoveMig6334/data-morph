"""Stage 3 — Claude Opus writes a conversion script from a metadata envelope.

Mirrors src/evaluation/teacher.py::_call_opus (same `claude -p` invocation), but
the model returns <analysis> + <script> rather than a converted file. The live
call is exercised only by opt-in tests; parsing/prompt building are pure.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKILL_REL_PATH = "skills/script_generation_teacher.md"

_ANALYSIS_RE = re.compile(r"<analysis>(.*?)</analysis>", re.DOTALL)
_SCRIPT_RE = re.compile(r"<script>(.*?)</script>", re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:python|py)?\s*\n(.*?)\n```$", re.DOTALL)


@dataclass
class ScriptResult:
    analysis: str
    script: str
    raw_output: str
    returncode: int
    stderr: str
    raw_payload: dict

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and bool(self.script)


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    m = _FENCE_RE.match(stripped)
    return m.group(1).strip() if m else stripped


def parse_teacher_output(text: str) -> tuple[str, str]:
    """Return (analysis, script). Script has any wrapping ```fence``` removed."""
    a = _ANALYSIS_RE.search(text)
    s = _SCRIPT_RE.search(text)
    analysis = a.group(1).strip() if a else ""
    script = _strip_fence(s.group(1)) if s else ""
    return analysis, script


def build_script_prompt(
    envelope: dict[str, Any],
    instruction: str,
    output_format: str,
    feedback: str | None = None,
) -> str:
    env_json = json.dumps(envelope, indent=2, default=str)
    fb = (
        f"\n\nYour previous attempt failed: {feedback}\n"
        f"Write a corrected <analysis> + <script>.\n"
        if feedback
        else ""
    )
    return (
        f"Read the instructions in {SKILL_REL_PATH}, then write a Python conversion script.\n\n"
        f"You are given the METADATA ENVELOPE of a source file (not the file itself):\n"
        f"```json\n{env_json}\n```\n\n"
        f"Task: {instruction}\n"
        f"Target output format: {output_format.upper()}.\n\n"
        f"The script must read the input file path from sys.argv[1] and write the converted "
        f"output to sys.argv[2], using only the Python standard library and pandas. Respond "
        f"with exactly an <analysis>...</analysis> block followed by a <script>...</script> "
        f"block. No prose, no code fences outside the script tags."
        f"{fb}"
    )


def call_script_teacher(
    envelope: dict[str, Any],
    instruction: str,
    output_format: str,
    *,
    timeout: int = 240,
    feedback: str | None = None,
) -> ScriptResult:
    """Run `claude -p --model opus` and parse <analysis> + <script> from the result."""
    prompt = build_script_prompt(envelope, instruction, output_format, feedback)
    cmd = [
        "claude", "-p", prompt,
        "--model", "opus",
        "--output-format", "json",
        "--allowedTools", "Read",
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        return ScriptResult("", "", "", proc.returncode, proc.stderr or "", {})
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return ScriptResult("", "", "", -1, f"decode error: {e}", {"stdout_head": proc.stdout[:500]})
    raw = payload.get("result", "") or ""
    analysis, script = parse_teacher_output(raw)
    return ScriptResult(analysis, script, raw, 0, proc.stderr or "", payload)
