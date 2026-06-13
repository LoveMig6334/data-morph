"""Shared primitives for the synthetic source-file generators (the oracle)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from faker import Faker

EXT_BY_FORMAT = {"csv": ".csv", "json": ".json", "txt": ".txt"}


@dataclass(frozen=True)
class GeneratedCase:
    """A single synthetic conversion case — both sides built from one seed."""

    use_case: str
    complexity: str
    input_format: str
    output_format: str
    input_text: str
    expected_text: str
    meta: dict[str, Any]


def make_faker(seed: int) -> Faker:
    """Return a Faker whose output is deterministic for the given seed."""
    fake = Faker()
    fake.seed_instance(seed)
    return fake


def write_case(case: GeneratedCase, dest_root: Path, case_name: str) -> Path:
    """Write input/expected/meta into dest_root/<use_case>/<case_name>/ (test_set layout)."""
    case_dir = dest_root / case.use_case / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / f"input{EXT_BY_FORMAT[case.input_format]}").write_text(
        case.input_text, encoding="utf-8"
    )
    (case_dir / f"expected{EXT_BY_FORMAT[case.output_format]}").write_text(
        case.expected_text, encoding="utf-8"
    )
    (case_dir / "meta.json").write_text(
        json.dumps(case.meta, indent=2) + "\n", encoding="utf-8"
    )
    return case_dir
