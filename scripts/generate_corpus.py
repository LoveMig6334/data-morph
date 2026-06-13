"""CLI: generate a reproducible synthetic corpus into data/raw.

Usage:
    uv run python scripts/generate_corpus.py --count 800 --dest data/raw
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datamorph.data.generators import (  # noqa: E402
    uc1_csv_to_json,
    uc2_json_to_csv,
    uc3_txt_log_to_csv,
    uc4_csv_to_txt_report,
    uc5_schema_migration,
)
from datamorph.data.generators.base import GeneratedCase, write_case  # noqa: E402

_GENERATORS = [
    uc1_csv_to_json.generate,
    uc2_json_to_csv.generate,
    uc3_txt_log_to_csv.generate,
    uc4_csv_to_txt_report.generate,
    uc5_schema_migration.generate,
]
_MIX = [("simple", 0.50), ("medium", 0.35), ("complex", 0.15)]
_TEST_SET = Path(__file__).resolve().parents[1] / "data" / "test_set"


def _test_set_input_hashes() -> set[str]:
    """MD5 of every test_set input file — used to reject leakage."""
    hashes: set[str] = set()
    if not _TEST_SET.exists():
        return hashes
    for p in _TEST_SET.glob("**/input.*"):
        hashes.add(hashlib.md5(p.read_bytes()).hexdigest())
    return hashes


def build_corpus(*, dest_root: Path, count: int, seed_base: int) -> dict[str, Any]:
    """Generate `count` cases into dest_root and return a manifest dict."""
    leak_hashes = _test_set_input_hashes()
    per_uc = count // len(_GENERATORS)
    cases: list[dict[str, Any]] = []
    complexity_counts = {"simple": 0, "medium": 0, "complex": 0}
    seed = seed_base
    idx = 0

    for gen in _GENERATORS:
        # Split this use case's quota across the difficulty mix (floor + remainder to last).
        _allocated = 0
        quotas: dict[str, int] = {}
        for i, (c, frac) in enumerate(_MIX):
            if i == len(_MIX) - 1:
                quotas[c] = per_uc - _allocated
            else:
                q = int(per_uc * frac)
                quotas[c] = q
                _allocated += q
        for complexity, n in quotas.items():
            for _ in range(n):
                case: GeneratedCase = gen(seed=seed, complexity=complexity)
                seed += 1
                in_hash = hashlib.md5(case.input_text.encode("utf-8")).hexdigest()
                if in_hash in leak_hashes:
                    continue  # skip leakage with the eval test_set
                case_name = f"gen_{idx:06d}"
                write_case(case, dest_root, case_name)
                cases.append(
                    {
                        "use_case": case.use_case,
                        "complexity": complexity,
                        "case_name": case_name,
                        "input_format": case.input_format,
                        "output_format": case.output_format,
                        "seed": case.meta.get("seed"),
                    }
                )
                complexity_counts[complexity] += 1
                idx += 1

    manifest = {
        "n_cases": len(cases),
        "complexity_counts": complexity_counts,
        "seed_base": seed_base,
        "cases": cases,
    }
    dest_root.mkdir(parents=True, exist_ok=True)
    (dest_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _main() -> int:
    parser = argparse.ArgumentParser(description="Generate the synthetic data-morph corpus.")
    parser.add_argument("--count", type=int, default=800, help="Total cases (default 800)")
    parser.add_argument("--dest", type=Path, default=Path("data/raw"), help="Output root")
    parser.add_argument("--seed-base", type=int, default=10_000, help="Base RNG seed")
    args = parser.parse_args()

    manifest = build_corpus(dest_root=args.dest, count=args.count, seed_base=args.seed_base)
    print(f"Wrote {manifest['n_cases']} cases to {args.dest}")
    print(f"  mix: {manifest['complexity_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
