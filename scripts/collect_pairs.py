"""CLI: generate verified training pairs from the corpus using Claude Opus.

Usage:
    uv run python scripts/collect_pairs.py --raw data/raw --interim data/interim --limit 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.collect import collect_corpus  # noqa: E402


def _main() -> int:
    parser = argparse.ArgumentParser(description="Collect verified envelope->script pairs via Opus.")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"), help="Corpus root")
    parser.add_argument("--interim", type=Path, default=Path("data/interim"), help="Output root")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N cases")
    parser.add_argument("--max-retries", type=int, default=3, help="Teacher retries per case")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cases that already have an accepted record (continue an interrupted run "
        "without re-spending teacher calls)",
    )
    args = parser.parse_args()

    summary = collect_corpus(
        args.raw,
        args.interim,
        max_retries=args.max_retries,
        limit=args.limit,
        resume=args.resume,
    )
    print(
        f"Processed {summary['n_attempted']} cases "
        f"(skipped {summary['n_skipped']} already-done) -> "
        f"{summary['n_accepted']} accepted "
        f"(accept rate {summary['accept_rate']:.1%}); "
        f"{summary['n_records_total']} records total in {args.interim}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
