"""CLI: turn verified interim records into train/val/test chat JSONL.

Usage:
    uv run python scripts/build_dataset.py --interim data/interim --processed data/processed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.format_pairs import split_records, to_chat_record  # noqa: E402

_SKIP = {"collect_manifest.json", "manifest.json"}


def build_dataset(interim_root: Path, processed_root: Path, *, seed: int = 0) -> dict[str, Any]:
    """Read interim records, split, and write {train,val,test}.jsonl of chat records."""
    records: list[dict[str, Any]] = []
    for p in sorted(interim_root.glob("*.json")):
        if p.name in _SKIP:
            continue
        records.append(json.loads(p.read_text(encoding="utf-8")))

    splits = split_records(records, seed=seed)
    processed_root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split, recs in splits.items():
        out = processed_root / f"{split}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(to_chat_record(rec), default=str) + "\n")
        counts[split] = len(recs)

    return {"n_records": len(records), "counts": counts}


def _main() -> int:
    parser = argparse.ArgumentParser(description="Build train/val/test chat JSONL from interim pairs.")
    parser.add_argument("--interim", type=Path, default=Path("data/interim"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    summary = build_dataset(args.interim, args.processed, seed=args.seed)
    print(f"{summary['n_records']} records -> {summary['counts']} in {args.processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
