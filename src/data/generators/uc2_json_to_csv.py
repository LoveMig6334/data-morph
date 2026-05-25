"""Generator: nested JSON (name + address object) -> flattened CSV (dot-notation columns)."""

from __future__ import annotations

import csv
import io
import json
import random

from .base import GeneratedCase, make_faker

_N = {"simple": 3, "medium": 8, "complex": 20}


def generate(seed: int, complexity: str) -> GeneratedCase:
    rng = random.Random(seed)
    fake = make_faker(seed)
    include_age = complexity != "simple"

    records: list[dict] = []
    for i in range(_N[complexity]):
        # Strip commas from city so the oracle CSV stays comma-clean.
        city = fake.city().replace(",", "")
        rec: dict = {
            "name": f"{fake.first_name()}{i}",
            "address": {"city": city, "zip": str(fake.postcode())},
        }
        if include_age:
            rec["age"] = rng.randint(18, 80)
        records.append(rec)

    input_text = json.dumps(records, indent=2) + "\n"

    cols = ["name", "address.city", "address.zip"] + (["age"] if include_age else [])
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(cols)
    for r in records:
        row = [r["name"], r["address"]["city"], r["address"]["zip"]]
        if include_age:
            row.append(r["age"])
        writer.writerow(row)
    expected_text = buf.getvalue()

    meta = {
        "use_case": "uc2_json_to_csv_flatten",
        "complexity": complexity,
        "input_format": "json",
        "output_format": "csv",
        "description": "Nested JSON records -> flattened CSV with dot-notation columns",
        "prompt_hint": (
            "Flatten nested keys with dot notation (address.city, address.zip). "
            "One CSV row per record. Header order: " + ",".join(cols) + "."
        ),
        "seed": seed,
    }
    return GeneratedCase(
        "uc2_json_to_csv_flatten", complexity, "json", "csv", input_text, expected_text, meta
    )
