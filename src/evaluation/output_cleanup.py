"""Heuristic cleanup for student-model output.

Gemma base will often wrap output in code fences or add prose preambles
despite explicit instructions. These steps are *opportunistic*: if a step
would damage content (e.g. unclosed fence), it skips. The raw text is also
stored on disk as `raw_actual.<fmt>` so any cleanup mistake is auditable.

Returns: (cleaned_text, list_of_steps_that_fired)
"""
from __future__ import annotations

import re


def clean_model_output(raw: str, output_format: str) -> tuple[str, list[str]]:
    applied: list[str] = []
    text = raw

    stripped = text.strip()
    if stripped != text:
        applied.append("strip_whitespace")
    text = stripped

    fenced = _try_strip_code_fence(text)
    if fenced is not None:
        text = fenced
        applied.append("strip_code_fence")

    pre_stripped = _try_strip_preamble(text, output_format)
    if pre_stripped is not None:
        text = pre_stripped
        applied.append("strip_preamble")

    if output_format == "json":
        trailing_stripped = _try_strip_trailing_prose_json(text)
        if trailing_stripped is not None:
            text = trailing_stripped
            applied.append("strip_trailing_prose")

    final = text.strip()
    if final != text and "strip_whitespace" not in applied:
        applied.append("strip_whitespace")
    return final, applied


_FENCE_OPEN = re.compile(r"^```([A-Za-z0-9_+\-]*)\s*\n", re.MULTILINE)
_FENCE_CLOSE = "\n```"


def _try_strip_code_fence(text: str) -> str | None:
    m = _FENCE_OPEN.match(text)
    if not m:
        return None
    body_start = m.end()
    close_idx = text.find(_FENCE_CLOSE, body_start)
    if close_idx == -1:
        return None  # unclosed — skip
    return text[body_start:close_idx]


def _try_strip_preamble(text: str, output_format: str) -> str | None:
    if output_format == "txt":
        return None
    lines = text.split("\n")
    if output_format == "json":
        for i, line in enumerate(lines):
            s = line.lstrip()
            if s.startswith("{") or s.startswith("["):
                if i == 0:
                    return None
                return "\n".join(lines[i:])
        return None
    if output_format == "csv":
        for i, line in enumerate(lines):
            if "," in line:
                if i == 0:
                    # First line already has a comma — likely a header; do not strip.
                    # (Conservative: a prose preamble containing a comma will not be stripped.)
                    return None
                return "\n".join(lines[i:])
        return None
    return None


def _try_strip_trailing_prose_json(text: str) -> str | None:
    # Find first opening bracket
    start = -1
    open_ch = ""
    for i, c in enumerate(text):
        if c == "{" or c == "[":
            start = i
            open_ch = c
            break
    if start == -1:
        return None
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    end = -1
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return None
    if end == len(text):
        return None  # nothing to strip
    if text[end:].strip() == "":
        return None  # only whitespace remains, not prose — defer to final strip_whitespace
    return text[:end]
