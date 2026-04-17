"""Claude Opus teacher — invokes the `claude -p` CLI as a subprocess.

The prompt tells Claude to Read the skill file each call; this is the agreed
substitute for a real Agent Skill (which is not supported in `-p` mode).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKILL_REL_PATH = "skills/file_conversion_teacher.md"


@dataclass
class TeacherResult:
    output: str
    raw_payload: dict
    returncode: int
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and bool(self.output)


def build_prompt(
    input_text: str,
    input_format: str,
    output_format: str,
    prompt_hint: str,
) -> str:
    """Assemble the prompt that references the skill file + embeds the input."""
    return (
        f"Read the instructions in {SKILL_REL_PATH}, then follow them to convert "
        f"the input below.\n\n"
        f"Conversion: {input_format.upper()} -> {output_format.upper()}\n"
        f"Task-specific notes: {prompt_hint}\n\n"
        f"Input (between the === markers):\n"
        f"===\n{input_text}\n===\n\n"
        f"Output the converted file content only. The first character of your "
        f"response must be the first character of the converted file. No prose, "
        f"no code fences, no markdown."
    )


def call_teacher(
    input_text: str,
    input_format: str,
    output_format: str,
    prompt_hint: str,
    timeout: int = 180,
) -> TeacherResult:
    prompt = build_prompt(input_text, input_format, output_format, prompt_hint)
    cmd = [
        "claude",
        "-p",
        prompt,
        "--model",
        "opus",
        "--output-format",
        "json",
        "--allowedTools",
        "Read",
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
        return TeacherResult(
            output="",
            raw_payload={},
            returncode=proc.returncode,
            stderr=proc.stderr or "",
        )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return TeacherResult(
            output="",
            raw_payload={"decode_error": str(e), "stdout_head": proc.stdout[:500]},
            returncode=-1,
            stderr=f"Could not decode claude -p JSON output: {e}",
        )
    output = payload.get("result", "") or ""
    return TeacherResult(
        output=output,
        raw_payload=payload,
        returncode=0,
        stderr=proc.stderr or "",
    )
