"""Stage 3+4+5 orchestrator: envelope -> teacher script -> sandbox -> verify.

The teacher is injected as `teacher_fn` so the loop is fully testable without
any API calls. `collect_corpus` is the batch driver used by the CLI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.data.envelope import extract_envelope
from src.data.generators.base import EXT_BY_FORMAT
from src.data.sandbox import run_script
from src.data.teacher_script import ScriptResult, call_script_teacher
from src.evaluation.metrics import score_all
from src.evaluation.runner import CaseSpec, discover_cases

CA_MIN = 0.95
TeacherFn = Callable[..., ScriptResult]


@dataclass
class PairResult:
    case_id: str
    use_case: str
    complexity: str
    input_format: str
    output_format: str
    accepted: bool = False
    scores: dict[str, float] = field(default_factory=dict)
    retries: int = 0
    error_kind: str = ""
    reason: str = ""
    envelope: dict[str, Any] = field(default_factory=dict)
    instruction: str = ""
    analysis: str = ""
    script: str = ""
    teacher_usage: dict[str, Any] | None = None  # Opus token usage (cost + W6 analysis)


def _passes(scores: dict[str, float]) -> bool:
    return (
        scores.get("format_validity", 0.0) == 1.0
        and scores.get("loadability", 0.0) == 1.0
        and scores.get("schema_compliance", 0.0) == 1.0
        and scores.get("content_accuracy", 0.0) >= CA_MIN
    )


def _failing_metrics(scores: dict[str, float]) -> list[str]:
    failing = []
    for k in ("format_validity", "loadability", "schema_compliance"):
        if scores.get(k, 0.0) < 1.0:
            failing.append(k)
    if scores.get("content_accuracy", 0.0) < CA_MIN:
        failing.append("content_accuracy")
    return failing


def collect_case(
    case: CaseSpec,
    *,
    teacher_fn: TeacherFn = call_script_teacher,
    max_retries: int = 3,
) -> PairResult:
    """Run the full teach->run->verify loop for one case, retrying with feedback."""
    meta = case.meta
    in_ext = EXT_BY_FORMAT[meta["input_format"]]
    out_ext = EXT_BY_FORMAT[meta["output_format"]]
    input_path = case.case_dir / f"input{in_ext}"

    envelope = extract_envelope(input_path, meta["input_format"])
    envelope.pop("file_path", None)  # don't leak local paths into training data
    instruction = meta.get("prompt_hint") or (
        f"Convert this {meta['input_format'].upper()} to {meta['output_format'].upper()}."
    )

    result = PairResult(
        case_id=case.case_id,
        use_case=meta["use_case"],
        complexity=meta["complexity"],
        input_format=meta["input_format"],
        output_format=meta["output_format"],
        envelope=envelope,
        instruction=instruction,
    )

    feedback: str | None = None
    for attempt in range(max_retries + 1):
        result.retries = attempt
        tr = teacher_fn(envelope, instruction, meta["output_format"], feedback=feedback)
        # Capture token usage from the latest teacher response (claude -p JSON
        # payload carries it). One-shot data — not recoverable after the run.
        usage = tr.raw_payload.get("usage") if tr.raw_payload else None
        if usage:
            result.teacher_usage = usage
        if not tr.ok:
            result.error_kind = "no_script"
            result.reason = f"teacher produced no <script> (stderr: {tr.stderr[:200]})"
            feedback = result.reason
            continue

        sr = run_script(tr.script, input_path, output_suffix=out_ext)
        if not sr.ok:
            result.error_kind = sr.error_kind
            result.reason = f"script {sr.error_kind}: {sr.stderr[:300]}"
            result.analysis, result.script = tr.analysis, tr.script
            feedback = result.reason
            continue

        scores = score_all(
            actual=sr.output_text,
            expected=case.expected_text,
            output_format=meta["output_format"],
            required_substrings=meta.get("required_substrings"),
        )
        result.scores = scores
        result.analysis, result.script = tr.analysis, tr.script
        if _passes(scores):
            result.accepted = True
            result.error_kind = "ok"
            result.reason = ""
            return result
        failing = _failing_metrics(scores)
        result.error_kind = "low_score"
        result.reason = f"output scored low on {failing}: {scores}"
        feedback = result.reason

    return result


def collect_corpus(
    raw_root: Path,
    interim_root: Path,
    *,
    teacher_fn: TeacherFn = call_script_teacher,
    max_retries: int = 3,
    limit: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run collect_case over every corpus case; write accepted records + a manifest.

    When ``resume`` is True, any case that already has an accepted record file in
    ``interim_root`` is skipped without calling the teacher, so an interrupted run
    (e.g. one stopped by a teacher usage limit) can be continued without
    re-spending Opus calls on pairs already collected.
    """
    cases = discover_cases(raw_root)
    if limit is not None:
        cases = cases[:limit]
    interim_root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    n_accepted = 0
    n_skipped = 0
    for case in cases:
        out_path = interim_root / f"{case.meta['use_case']}__{case.case_dir.name}.json"
        if resume and out_path.exists():
            n_skipped += 1
            manifest.append(
                {
                    "case_id": case.case_id,
                    "use_case": case.meta["use_case"],
                    "accepted": True,
                    "error_kind": "skipped_existing",
                    "retries": 0,
                    "scores": {},
                    "reason": "",
                }
            )
            continue
        res = collect_case(case, teacher_fn=teacher_fn, max_retries=max_retries)
        if res.accepted:
            n_accepted += 1
            record = {
                "case_id": res.case_id,
                "use_case": res.use_case,
                "complexity": res.complexity,
                "input_format": res.input_format,
                "output_format": res.output_format,
                "envelope": res.envelope,
                "instruction": res.instruction,
                "analysis": res.analysis,
                "script": res.script,
                "scores": res.scores,
                "retries": res.retries,
                "teacher_usage": res.teacher_usage,
            }
            out_path = interim_root / f"{res.use_case}__{case.case_dir.name}.json"
            out_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        manifest.append(
            {
                "case_id": res.case_id,
                "use_case": res.use_case,
                "accepted": res.accepted,
                "error_kind": res.error_kind,
                "retries": res.retries,
                "scores": res.scores,
                "reason": res.reason if not res.accepted else "",
            }
        )

    n_attempted = len(cases) - n_skipped
    summary = {
        "n_cases": len(cases),
        "n_skipped": n_skipped,
        "n_attempted": n_attempted,
        "n_accepted": n_accepted,
        "n_records_total": n_accepted + n_skipped,
        # accept_rate is over cases actually attempted this run (skipped ones
        # were already accepted), so a resumed run's rate stays meaningful.
        "accept_rate": round(n_accepted / n_attempted, 3) if n_attempted else 0.0,
        "results": manifest,
    }
    (interim_root / "collect_manifest.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary
