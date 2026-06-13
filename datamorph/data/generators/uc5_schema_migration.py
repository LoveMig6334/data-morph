"""Generator: JSON schema migration — rename user_* keys to bare names."""

from __future__ import annotations

import json
import random

from .base import GeneratedCase, make_faker

_N = {"simple": 3, "medium": 8, "complex": 20}


def generate(seed: int, complexity: str) -> GeneratedCase:
    rng = random.Random(seed)
    fake = make_faker(seed)
    include_age = complexity != "simple"

    src: list[dict] = []
    dst: list[dict] = []
    for i in range(_N[complexity]):
        name = f"{fake.first_name()}{i}"
        email = f"{name.lower()}@example.com"
        s: dict = {"user_name": name, "user_email": email}
        d: dict = {"name": name, "email": email}
        if include_age:
            age = rng.randint(18, 80)
            s["user_age"] = age
            d["age"] = age
        src.append(s)
        dst.append(d)

    input_text = json.dumps(src, indent=2) + "\n"
    expected_text = json.dumps(dst, indent=2) + "\n"

    meta = {
        "use_case": "uc5_schema_migration",
        "complexity": complexity,
        "input_format": "json",
        "output_format": "json",
        "description": "JSON schema migration — rename user_* keys to bare names",
        "prompt_hint": (
            "Rename keys on every record: user_name->name, user_email->email, "
            "user_age->age. Preserve values and record order."
        ),
        "seed": seed,
    }
    return GeneratedCase(
        "uc5_schema_migration", complexity, "json", "json", input_text, expected_text, meta
    )
