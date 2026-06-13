"""Orchestrates the baseline evaluation across every test case."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .metrics import score_all
from .teacher import call_teacher

EXT_BY_FORMAT = {"csv": ".csv", "json": ".json", "txt": ".txt"}


@dataclass
class CaseSpec:
    case_dir: Path
    meta: dict
    input_text: str
    expected_text: str

    @property
    def case_id(self) -> str:
        return f"{self.case_dir.parent.name}/{self.case_dir.name}"


@dataclass
class CaseResult:
    case_id: str
    use_case: str
    complexity: str
    input_format: str
    output_format: str
    scores: dict[str, float] = field(default_factory=dict)
    output_preview: str = ""
    ok: bool = False
    error: str | None = None
    elapsed_sec: float = 0.0


_COMPLEXITY_ORDER = {"simple": 0, "medium": 1, "complex": 2}


def _case_sort_key(case_dir: Path) -> tuple:
    # Sort by use-case dir, then by complexity (simple -> medium -> complex),
    # then by case name. Avoids alphabetical mediums running before simples.
    name = case_dir.name
    complexity = name.split("_")[0]
    return (
        case_dir.parent.name,
        _COMPLEXITY_ORDER.get(complexity, 99),
        name,
    )


def discover_cases(test_root: Path) -> list[CaseSpec]:
    cases: list[CaseSpec] = []
    case_dirs = sorted(
        (d for d in test_root.glob("*/*/") if d.is_dir()),
        key=_case_sort_key,
    )
    for case_dir in case_dirs:
        if not case_dir.is_dir():
            continue
        meta_path = case_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        in_ext = EXT_BY_FORMAT[meta["input_format"]]
        out_ext = EXT_BY_FORMAT[meta["output_format"]]
        input_path = case_dir / f"input{in_ext}"
        expected_path = case_dir / f"expected{out_ext}"
        if not input_path.exists() or not expected_path.exists():
            continue
        cases.append(
            CaseSpec(
                case_dir=case_dir,
                meta=meta,
                input_text=input_path.read_text(encoding="utf-8"),
                expected_text=expected_path.read_text(encoding="utf-8"),
            )
        )
    return cases


def run_case(case: CaseSpec, outputs_dir: Path, model: str = "opus") -> CaseResult:
    meta = case.meta
    result = CaseResult(
        case_id=case.case_id,
        use_case=meta["use_case"],
        complexity=meta["complexity"],
        input_format=meta["input_format"],
        output_format=meta["output_format"],
    )
    started = time.time()
    teacher_result = call_teacher(
        input_text=case.input_text,
        input_format=meta["input_format"],
        output_format=meta["output_format"],
        prompt_hint=meta.get("prompt_hint", ""),
        model=model,
    )
    result.elapsed_sec = round(time.time() - started, 2)

    # Persist the raw teacher output even on failure — useful for error analysis.
    case_out_dir = outputs_dir / case.case_dir.parent.name / case.case_dir.name
    case_out_dir.mkdir(parents=True, exist_ok=True)
    out_ext = EXT_BY_FORMAT[meta["output_format"]]

    # Cleaned (or for Opus, unchanged) output — what the metrics score.
    (case_out_dir / f"actual{out_ext}").write_text(
        teacher_result.output, encoding="utf-8"
    )
    # For Gemma: also persist the raw pre-cleanup output for audit.
    if model == "gemma":
        (case_out_dir / f"raw_actual{out_ext}").write_text(
            teacher_result.raw_output, encoding="utf-8"
        )

    # Per-backend metadata; filename kept as teacher_meta.json for artefact parity.
    if model == "opus":
        meta_payload: dict[str, Any] = {
            "returncode": teacher_result.returncode,
            "stderr": teacher_result.stderr[:1000],
            "usage": teacher_result.raw_payload.get("usage"),
            "session_id": teacher_result.raw_payload.get("session_id"),
            "elapsed_sec": result.elapsed_sec,
        }
    else:  # gemma
        gm = teacher_result.gemma_meta or {}
        raw_bytes = len(teacher_result.raw_output.encode("utf-8"))
        clean_bytes = len(teacher_result.output.encode("utf-8"))
        meta_payload = {
            "model_id": gm.get("model_id"),
            "n_prompt_tokens": gm.get("n_prompt_tokens"),
            "n_generated_tokens": gm.get("n_generated_tokens"),
            "tokens_per_sec": gm.get("tokens_per_sec"),
            "elapsed_sec": gm.get("elapsed_sec", result.elapsed_sec),
            "truncated": gm.get("truncated", False),
            "cleanup_applied": teacher_result.cleanup_applied,
            "raw_size_bytes": raw_bytes,
            "cleaned_size_bytes": clean_bytes,
            "stderr": teacher_result.stderr[:1000] or None,
        }
    (case_out_dir / "teacher_meta.json").write_text(
        json.dumps(meta_payload, indent=2),
        encoding="utf-8",
    )

    if not teacher_result.ok:
        result.ok = False
        result.error = teacher_result.stderr[:500] or "teacher returned empty output"
        result.scores = {
            "format_validity": 0.0,
            "schema_compliance": 0.0,
            "loadability": 0.0,
            "content_accuracy": 0.0,
        }
        return result

    result.ok = True
    result.output_preview = teacher_result.output[:200]
    result.scores = score_all(
        actual=teacher_result.output,
        expected=case.expected_text,
        output_format=meta["output_format"],
        required_substrings=meta.get("required_substrings"),
    )
    return result


def aggregate(results: list[CaseResult]) -> dict[str, Any]:
    if not results:
        return {}
    metric_keys = [
        "format_validity",
        "schema_compliance",
        "loadability",
        "content_accuracy",
    ]
    overall = {
        k: round(sum(r.scores.get(k, 0.0) for r in results) / len(results), 3)
        for k in metric_keys
    }

    by_uc: dict[str, dict[str, Any]] = {}
    for r in results:
        bucket = by_uc.setdefault(r.use_case, {"n": 0, **{k: 0.0 for k in metric_keys}})
        bucket["n"] += 1
        for k in metric_keys:
            bucket[k] += r.scores.get(k, 0.0)
    for uc, bucket in by_uc.items():
        n = bucket["n"]
        for k in metric_keys:
            bucket[k] = round(bucket[k] / n, 3)

    by_complexity: dict[str, dict[str, Any]] = {}
    for r in results:
        bucket = by_complexity.setdefault(
            r.complexity, {"n": 0, **{k: 0.0 for k in metric_keys}}
        )
        bucket["n"] += 1
        for k in metric_keys:
            bucket[k] += r.scores.get(k, 0.0)
    for c, bucket in by_complexity.items():
        n = bucket["n"]
        for k in metric_keys:
            bucket[k] = round(bucket[k] / n, 3)

    return {
        "overall": overall,
        "by_use_case": by_uc,
        "by_complexity": by_complexity,
        "n_cases": len(results),
        "n_inference_errors": sum(1 for r in results if not r.ok),
    }
