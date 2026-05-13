"""CLI entry: run the baseline across every test case.

Usage (from project root):
    python scripts/run_baseline.py
    python scripts/run_baseline.py --model gemma  # student baseline
    python scripts/run_baseline.py --limit 3      # quick smoke-test
    python scripts/run_baseline.py --only uc1     # filter by use case prefix
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.runner import (  # noqa: E402
    aggregate,
    discover_cases,
    run_case,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-root", default="data/test_set")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N cases (for smoke-testing).",
    )
    parser.add_argument(
        "--only", default=None, help="Substring filter on use-case directory name."
    )
    parser.add_argument(
        "--model",
        choices=["opus", "gemma"],
        default="opus",
        help="Inference backend (default: opus, the W2 teacher baseline).",
    )
    args = parser.parse_args()

    test_root = PROJECT_ROOT / args.test_root
    cases = discover_cases(test_root)
    if args.only:
        cases = [c for c in cases if args.only in c.case_dir.parent.name]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print(f"No cases discovered under {test_root}", file=sys.stderr)
        return 1

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = PROJECT_ROOT / args.results_dir / f"baseline_{args.model}_{stamp}"
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(cases)} cases (model={args.model}). Results -> {run_dir}")
    per_case = []
    for idx, case in enumerate(cases, 1):
        print(f"  [{idx:2d}/{len(cases)}] {case.case_id} ... ", end="", flush=True)
        result = run_case(case, outputs_dir, model=args.model)
        status = "OK" if result.ok else "FAIL"
        mv = result.scores.get("format_validity", 0.0)
        ca = result.scores.get("content_accuracy", 0.0)
        print(f"{status}  fv={mv:.2f} ca={ca:.2f}  ({result.elapsed_sec}s)")
        per_case.append(result)

    backend = "claude-cli" if args.model == "opus" else "mlx"
    role = "teacher" if args.model == "opus" else "student-base"
    model_label = (
        "claude -p --model opus (Claude Code subscription)"
        if args.model == "opus"
        else "mlx-community/gemma-2-2b-it (no fine-tune)"
    )
    summary = {
        "run_id": f"baseline_{args.model}_{stamp}",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": model_label,
        "backend": backend,
        "role": role,
        "aggregate": aggregate(per_case),
        "cases": [
            {
                "case_id": r.case_id,
                "use_case": r.use_case,
                "complexity": r.complexity,
                "input_format": r.input_format,
                "output_format": r.output_format,
                "ok": r.ok,
                "error": r.error,
                "elapsed_sec": r.elapsed_sec,
                "scores": r.scores,
                "output_preview": r.output_preview,
            }
            for r in per_case
        ],
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    agg = summary["aggregate"]
    print("=== Aggregate ===")
    n_err = agg.get("n_inference_errors", agg.get("n_teacher_errors", 0))
    print(f"  n_cases: {agg['n_cases']}   inference_errors: {n_err}")
    print("  overall:")
    for k, v in agg["overall"].items():
        print(f"    {k:18s} {v:.3f}")
    print("  by complexity:")
    for c, s in agg["by_complexity"].items():
        print(
            f"    {c:8s} n={s['n']}  "
            f"fv={s['format_validity']:.2f}  "
            f"sc={s['schema_compliance']:.2f}  "
            f"ld={s['loadability']:.2f}  "
            f"ca={s['content_accuracy']:.2f}"
        )
    print(f"\nFull summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
