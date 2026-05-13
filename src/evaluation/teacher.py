"""Inference backends for the W2 baseline pipeline.

`model="opus"` runs the original `claude -p --model opus` subprocess (W2 teacher).
`model="gemma"` runs Gemma 2 2B IT via MLX in-process (student baseline, pre-fine-tune).

Filename `teacher.py` is kept as a misnomer to avoid breaking existing imports;
the module now hosts both teacher and student inference paths.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKILL_REL_PATH = "skills/file_conversion_teacher.md"


@dataclass
class TeacherResult:
    output: str
    raw_payload: dict
    returncode: int
    stderr: str
    raw_output: str = ""          # pre-cleanup; equals `output` for Opus path
    gemma_meta: dict | None = None  # Gemma-only inference metadata
    cleanup_applied: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and bool(self.output)


def build_user_prompt(
    input_text: str,
    input_format: str,
    output_format: str,
    prompt_hint: str,
    reference_skill: bool,
) -> str:
    """Assemble the user-role body.

    `reference_skill=True` (Opus): instruct the model to Read the skill file.
    `reference_skill=False` (Gemma): omit — the skill content is concatenated
    in front of this body by the caller, since Gemma 2's chat template has
    no `system` role.
    """
    preamble = (
        f"Read the instructions in {SKILL_REL_PATH}, then follow them to convert "
        f"the input below.\n\n"
        if reference_skill
        else ""
    )
    return (
        f"{preamble}"
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
    model: str = "opus",
) -> TeacherResult:
    if model == "opus":
        return _call_opus(input_text, input_format, output_format, prompt_hint, timeout)
    if model == "gemma":
        return _call_gemma(input_text, input_format, output_format, prompt_hint)
    raise ValueError(f"Unknown model: {model!r} (expected 'opus' or 'gemma')")


def _call_opus(
    input_text: str,
    input_format: str,
    output_format: str,
    prompt_hint: str,
    timeout: int,
) -> TeacherResult:
    prompt = build_user_prompt(
        input_text, input_format, output_format, prompt_hint, reference_skill=True
    )
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
        raw_output=output,  # Opus output is not cleaned
        raw_payload=payload,
        returncode=0,
        stderr=proc.stderr or "",
    )


_SKILL_CACHE: dict[str, str] = {}


def _load_skill_text() -> str:
    if "text" not in _SKILL_CACHE:
        skill_path = PROJECT_ROOT / SKILL_REL_PATH
        _SKILL_CACHE["text"] = skill_path.read_text(encoding="utf-8")
    return _SKILL_CACHE["text"]


def _call_gemma(
    input_text: str,
    input_format: str,
    output_format: str,
    prompt_hint: str,
) -> TeacherResult:
    from src.evaluation.output_cleanup import clean_model_output
    from src.models.gemma_mlx import generate as mlx_generate

    skill = _load_skill_text()
    user_body = build_user_prompt(
        input_text, input_format, output_format, prompt_hint, reference_skill=False
    )
    # Gemma 2's chat template does not support a `system` role — fold the
    # skill text into the user message, separated from the task instructions
    # by a clear delimiter.
    combined_user = f"{skill}\n\n---\n\n{user_body}"
    messages = [{"role": "user", "content": combined_user}]
    try:
        gen = mlx_generate(messages)
    except Exception as e:
        return TeacherResult(
            output="",
            raw_payload={},
            returncode=-1,
            stderr=f"gemma_mlx.generate raised: {e!r}",
        )

    cleaned, applied = clean_model_output(gen.text, output_format)
    return TeacherResult(
        output=cleaned,
        raw_output=gen.text,
        raw_payload={"model_id": gen.model_id},
        returncode=0,
        stderr="",
        cleanup_applied=applied,
        gemma_meta={
            "model_id": gen.model_id,
            "n_prompt_tokens": gen.n_prompt_tokens,
            "n_generated_tokens": gen.n_generated_tokens,
            "tokens_per_sec": gen.tokens_per_sec,
            "elapsed_sec": gen.elapsed_sec,
            "truncated": gen.truncated,
        },
    )
