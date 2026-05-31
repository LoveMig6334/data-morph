"""Zero-shot base-student baseline on the NEW pipeline.

Runs the un-fine-tuned local Gemma through the same Stage 3->4->5 path the
fine-tuned student will use (envelope -> script -> sandbox -> the four metrics),
over the held-out **test** split. This is the apples-to-apples "before fine-tuning"
number — unlike the 2026-05-13 baselines, which measured the old direct-conversion
task on the 15-case W2 set.

The test cases are reproduced deterministically from the seed-0 split of the 800
verified pairs (same split that produced data/processed/test.jsonl), then mapped
back to their source dirs in data/raw so the sandbox has the real input + expected
output. Single attempt per case (no retries).

Usage:
    uv run python scripts/run_pipeline_baseline.py            # all 70 test cases
    uv run python scripts/run_pipeline_baseline.py --limit 5  # quick smoke
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.collect import collect_case  # noqa: E402
from src.evaluation.runner import discover_cases  # noqa: E402
from src.features.format_pairs import split_records  # noqa: E402
from src.models.gemma_script_teacher import call_gemma_script_teacher  # noqa: E402

METRICS = ["format_validity", "schema_compliance", "loadability", "content_accuracy"]
CX_ORDER = ["simple", "medium", "complex"]


def test_case_ids(interim_root: Path, seed: int = 0) -> set[str]:
    """Reproduce the held-out test split (by case_id) from the verified pairs."""
    recs = [
        {"case_id": json.loads(f.read_text(encoding="utf-8"))["case_id"]}
        for f in sorted(interim_root.glob("*__gen_*.json"))
    ]
    return {r["case_id"] for r in split_records(recs, seed=seed)["test"]}


def _block(rows: list[dict]) -> dict:
    n = len(rows)
    out: dict = {"n": n}
    for m in METRICS:
        out[m] = round(sum(r["scores"].get(m, 0.0) for r in rows) / n, 3) if n else 0.0
    return out


def aggregate(rows: list[dict]) -> dict:
    by_uc: dict[str, list] = defaultdict(list)
    by_cx: dict[str, list] = defaultdict(list)
    for r in rows:
        by_uc[r["use_case"]].append(r)
        by_cx[r["complexity"]].append(r)
    overall = {m: round(sum(r["scores"].get(m, 0.0) for r in rows) / len(rows), 3) for m in METRICS}
    return {
        "overall": overall,
        "by_use_case": {k: _block(v) for k, v in sorted(by_uc.items())},
        "by_complexity": {k: _block(by_cx[k]) for k in CX_ORDER if k in by_cx},
        "n_cases": len(rows),
        "n_accepted": sum(1 for r in rows if r["accepted"]),
        "n_inference_errors": sum(1 for r in rows if r["error_kind"] == "no_script"),
        "error_kinds": dict(_count(r["error_kind"] for r in rows)),
        "n_cases_with_retries": sum(1 for r in rows if r.get("retries", 0) > 0),
        "total_retries": sum(r.get("retries", 0) for r in rows),
    }


def _count(it):
    c: dict = defaultdict(int)
    for x in it:
        c[x] += 1
    return c


def resolve_adapter(arg: str | None) -> tuple[str | None, str | None]:
    """Resolve --adapter to a directory mlx_vlm can load (with adapters.safetensors).

    Accepts either an adapter directory or a single checkpoint file
    (e.g. 0001000_adapters.safetensors). A checkpoint file is staged into a temp
    dir as adapters.safetensors alongside the sibling adapter_config.json.

    Returns (load_dir, label). The temp dir (if any) lives for the process.
    """
    if not arg:
        return None, None
    p = Path(arg)
    if p.is_dir():
        if not (p / "adapters.safetensors").exists():
            raise FileNotFoundError(f"{p} has no adapters.safetensors")
        return str(p), p.name
    if p.is_file():
        cfg = p.parent / "adapter_config.json"
        if not cfg.exists():
            raise FileNotFoundError(f"no adapter_config.json next to {p}")
        staged = Path(tempfile.mkdtemp(prefix="adapter_"))
        shutil.copy(p, staged / "adapters.safetensors")
        shutil.copy(cfg, staged / "adapter_config.json")
        return str(staged), p.stem  # label by checkpoint name
    raise FileNotFoundError(f"adapter path not found: {arg}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interim", default="data/interim")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default=None, help="output dir (default auto-named under results/)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="filter to case_ids starting with this prefix (e.g. uc4)")
    ap.add_argument(
        "--adapter",
        default=None,
        help="LoRA adapter dir or checkpoint .safetensors to evaluate the fine-tuned student "
        "(omit for the base-model baseline)",
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=0,
        help="max retries with error feedback per case (0 = one-shot; production uses 3)",
    )
    args = ap.parse_args()

    adapter_dir, adapter_label = resolve_adapter(args.adapter)
    if adapter_dir:
        from src.models import gemma_mlx

        gemma_mlx.use_adapter(adapter_dir)

    interim = PROJECT_ROOT / args.interim
    raw = PROJECT_ROOT / args.raw
    ids = test_case_ids(interim)
    cases = [c for c in discover_cases(raw) if c.case_id in ids]
    if args.only:
        cases = [c for c in cases if c.case_id.startswith(args.only)]
    if args.limit:
        cases = cases[: args.limit]

    who = f"fine-tuned ({adapter_label})" if adapter_dir else "base-student"
    mode = "one-shot" if args.retries == 0 else f"retry<={args.retries} w/ error feedback"
    print(f"new-pipeline {who} eval ({mode}): {len(cases)} held-out test cases")
    print("(Gemma loads on the first case)\n")

    rows: list[dict] = []
    for i, case in enumerate(cases, 1):
        res = collect_case(case, teacher_fn=call_gemma_script_teacher, max_retries=args.retries)
        sc = res.scores
        rows.append(
            {
                "case_id": res.case_id,
                "use_case": res.use_case,
                "complexity": res.complexity,
                "input_format": res.input_format,
                "output_format": res.output_format,
                "accepted": res.accepted,
                "error_kind": res.error_kind,
                "retries": res.retries,
                "scores": sc,
                "reason": res.reason[:300],
            }
        )
        flag = "OK" if res.accepted else f"x:{res.error_kind}"
        rtag = f" (retries={res.retries})" if res.retries else ""
        print(
            f"[{i:>2}/{len(cases)}] {flag:14} {res.case_id:42} "
            f"fv={sc.get('format_validity', 0):.0f} sc={sc.get('schema_compliance', 0):.0f} "
            f"ld={sc.get('loadability', 0):.0f} ca={sc.get('content_accuracy', 0):.2f}{rtag}",
            flush=True,
        )

    agg = aggregate(rows)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    if args.out:
        out = Path(args.out)
    elif adapter_dir:
        out = PROJECT_ROOT / "results" / f"eval_finetuned_{adapter_label}_{stamp}"
    else:
        out = PROJECT_ROOT / "results" / f"baseline_newpipeline_gemma_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    retry_desc = "one-shot, no retries" if args.retries == 0 else f"retry<={args.retries} w/ error feedback"
    summary = {
        "run_id": out.name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": os.environ.get("GEMMA_MLX_MODEL", "models/gemma-4-e2b-it-bf16"),
        "adapter": args.adapter,
        "max_retries": args.retries,
        "role": "student_finetuned_newpipeline" if adapter_dir else "student_base_newpipeline",
        "backend": "mlx_vlm",
        "pipeline": f"envelope->script->sandbox->metrics ({retry_desc}, skill in-context)",
        "eval_set": "held-out test split (seed=0, test_frac=0.1) reproduced from data/interim",
        "aggregate": agg,
        "cases": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"\n=== overall ({who}, {mode}) ===")
    for m in METRICS:
        print(f"  {m:18} {agg['overall'][m]:.3f}")
    print(f"  accepted (all-pass)  {agg['n_accepted']}/{agg['n_cases']}")
    print(f"  error kinds          {agg['error_kinds']}")
    print(f"  cases using retries  {agg['n_cases_with_retries']} (total {agg['total_retries']} retries)")
    print(f"\nwrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
