"""Public API: convert a file between formats with the distilled student model.

``convert_file`` runs the production pipeline — extract a metadata envelope, have
the student write a Python conversion script, run it in a sandbox, and validate the
output — retrying on failures up to ``max_retries``. The model never sees the full
source file, only its envelope.

    from datamorph import convert_file
    result = convert_file("contacts.csv", "contacts.json")
    print(result.accepted, result.output_path)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from datamorph.data.envelope import extract_envelope
from datamorph.data.sandbox import run_script
from datamorph.evaluation.metrics import format_validity, loadability
from datamorph.model import resolve_model

# Self-contained format <-> extension map (kept here so the inference path does
# not import the data-generation package, which pulls in faker).
EXT = {"csv": ".csv", "json": ".json", "txt": ".txt"}
_FMT_BY_EXT = {ext: fmt for fmt, ext in EXT.items()}
SUPPORTED_FORMATS = tuple(EXT)


class TeacherFn(Protocol):
    """Signature of the script author (the student model, or a test stub)."""

    def __call__(
        self, envelope: dict[str, Any], instruction: str, output_format: str,
        *, feedback: str | None = ...,
    ) -> Any: ...


@dataclass
class ConversionResult:
    """Outcome of a single ``convert_file`` call."""

    output_text: str
    input_format: str
    output_format: str
    script: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    accepted: bool = False
    retries: int = 0
    error: str | None = None
    output_path: Path | None = None


def _detect_format(path: Path, explicit: str | None, role: str) -> str:
    fmt = explicit.lower() if explicit else _FMT_BY_EXT.get(path.suffix.lower())
    if fmt not in EXT:
        raise ValueError(
            f"Unsupported or undetected {role} format for {path.name!r}; pass "
            f"{role}_format=<one of {SUPPORTED_FORMATS}>."
        )
    return fmt


def _default_teacher_fn(model: str | None) -> Callable:
    """Select the model and return the real student script author."""
    from datamorph.models import gemma_mlx
    from datamorph.models.gemma_script_teacher import call_gemma_script_teacher

    gemma_mlx.use_model(resolve_model(model), text_only=True)
    return call_gemma_script_teacher


def convert_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    input_format: str | None = None,
    output_format: str | None = None,
    instruction: str | None = None,
    max_retries: int = 3,
    model: str | None = None,
    teacher_fn: TeacherFn | None = None,
) -> ConversionResult:
    """Convert ``input_path`` to the target format, optionally writing ``output_path``.

    Formats are auto-detected from file extensions when not given explicitly. The
    pipeline retries up to ``max_retries`` times with error feedback. ``teacher_fn``
    can be injected to run the pipeline without the model (used in tests).
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    in_fmt = _detect_format(input_path, input_format, "input")
    if output_format:
        out_fmt = output_format.lower()
        if out_fmt not in EXT:
            raise ValueError(
                f"Unsupported output_format {output_format!r}; one of {SUPPORTED_FORMATS}."
            )
    elif output_path is not None:
        out_fmt = _detect_format(Path(output_path), None, "output")
    else:
        raise ValueError("Provide output_format=, or an output_path with a known extension.")

    if teacher_fn is None:
        teacher_fn = _default_teacher_fn(model)

    envelope = extract_envelope(input_path, in_fmt)
    envelope.pop("file_path", None)  # never leak local paths
    instruction = instruction or f"Convert this {in_fmt.upper()} to {out_fmt.upper()}."
    out_suffix = EXT[out_fmt]

    feedback: str | None = None
    result = ConversionResult("", in_fmt, out_fmt, error="not_run")
    for attempt in range(max_retries + 1):
        tr = teacher_fn(envelope, instruction, out_fmt, feedback=feedback)
        if not tr.ok:
            result = ConversionResult("", in_fmt, out_fmt, script=tr.script,
                                      retries=attempt, error="no_script")
            feedback = "Your previous response had no <script> block. Output one."
            continue
        sr = run_script(tr.script, input_path, output_suffix=out_suffix)
        if not sr.ok:
            result = ConversionResult(sr.output_text, in_fmt, out_fmt, script=tr.script,
                                      retries=attempt, error=sr.error_kind)
            feedback = f"The script failed ({sr.error_kind}): {sr.stderr[:300]}. Fix it."
            continue
        out = sr.output_text
        scores = {
            "format_validity": format_validity(out, out_fmt),
            "loadability": loadability(out, out_fmt),
        }
        accepted = scores["format_validity"] == 1.0 and scores["loadability"] == 1.0
        result = ConversionResult(out, in_fmt, out_fmt, script=tr.script, scores=scores,
                                  accepted=accepted, retries=attempt, error=None)
        if accepted:
            break
        feedback = f"Output was not valid {out_fmt.upper()} (scores={scores}). Fix the script."

    if output_path is not None and result.output_text:
        output_path = Path(output_path)
        output_path.write_text(result.output_text, encoding="utf-8")
        result.output_path = output_path
    return result
