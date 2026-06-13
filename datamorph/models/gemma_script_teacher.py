"""Gemma script-generation 'teacher' — the un-fine-tuned student on the new pipeline.

Drop-in replacement for ``datamorph.data.teacher_script.call_script_teacher`` (same
``ScriptResult`` contract and ``(envelope, instruction, output_format, *, feedback)``
signature), but the script author is the local Gemma model via MLX instead of Opus.

Used to establish the **zero-shot base-student baseline** on the Stage-3..5 pipeline
(envelope -> script -> sandbox -> metrics), before any LoRA fine-tuning. The base
model has not learned the skill, so — like the W2 Gemma baseline — the skill text is
folded into the prompt (Gemma's chat template has no system role and it cannot Read
files the way the Opus CLI can).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datamorph.data.teacher_script import ScriptResult, parse_teacher_output

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKILL_REL_PATH = "skills/script_generation_teacher.md"

_SKILL_CACHE: dict[str, str] = {}


def _skill_text() -> str:
    if "text" not in _SKILL_CACHE:
        _SKILL_CACHE["text"] = (PROJECT_ROOT / SKILL_REL_PATH).read_text(encoding="utf-8")
    return _SKILL_CACHE["text"]


def build_gemma_prompt(
    envelope: dict[str, Any],
    instruction: str,
    output_format: str,
    feedback: str | None = None,
) -> str:
    """Skill text + the inference-style task (envelope + instruction + output contract).

    The task half mirrors ``format_pairs.to_chat_record`` (what the fine-tuned model
    will see at inference); the skill text is prepended so the *un-trained* base model
    has the instructions in-context.
    """
    env_json = json.dumps(envelope, indent=2, default=str)
    fb = (
        f"\n\nYour previous attempt failed: {feedback}\nWrite a corrected response.\n"
        if feedback
        else ""
    )
    return (
        f"{_skill_text()}\n\n---\n\n"
        f"Metadata envelope:\n```json\n{env_json}\n```\n\n"
        f"Task: {instruction}\n"
        f"Target output format: {output_format.upper()}.\n"
        f"Write a Python conversion script that reads the input file path from "
        f"sys.argv[1] and writes the converted output to sys.argv[2], using only the "
        f"Python standard library and pandas. Respond with exactly an "
        f"<analysis>...</analysis> block followed by a <script>...</script> block. "
        f"No prose outside the tags."
        f"{fb}"
    )


def call_gemma_script_teacher(
    envelope: dict[str, Any],
    instruction: str,
    output_format: str,
    *,
    feedback: str | None = None,
    max_tokens: int = 4096,
) -> ScriptResult:
    """Generate <analysis> + <script> with the local Gemma model; parse into a ScriptResult."""
    from datamorph.models.gemma_mlx import generate as mlx_generate  # lazy: keep MLX import optional

    prompt = build_gemma_prompt(envelope, instruction, output_format, feedback)
    try:
        gen = mlx_generate([{"role": "user", "content": prompt}], max_tokens=max_tokens)
    except Exception as e:  # MLX / generation failure -> looks like "no script"
        return ScriptResult("", "", "", -1, f"gemma generate raised: {e!r}", {})

    analysis, script = parse_teacher_output(gen.text)
    return ScriptResult(
        analysis=analysis,
        script=script,
        raw_output=gen.text,
        returncode=0,
        stderr="",
        raw_payload={
            "model_id": gen.model_id,
            "gemma_meta": {
                "n_prompt_tokens": gen.n_prompt_tokens,
                "n_generated_tokens": gen.n_generated_tokens,
                "tokens_per_sec": gen.tokens_per_sec,
                "elapsed_sec": gen.elapsed_sec,
                "truncated": gen.truncated,
            },
        },
    )
