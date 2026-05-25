"""Generator: regional sales CSV -> human-readable TXT report with totals."""

from __future__ import annotations

import csv
import io
import random

from .base import GeneratedCase

_POOL = [
    "North", "South", "East", "West", "Central",
    "Northeast", "Southwest", "Northwest", "Southeast",
]
_N = {"simple": 4, "medium": 6, "complex": 9}


def generate(seed: int, complexity: str) -> GeneratedCase:
    rng = random.Random(seed)
    regions = _POOL[: _N[complexity]]
    data: list[tuple[str, int, int]] = [
        (reg, rng.randint(1000, 20000), rng.randint(5, 50)) for reg in regions
    ]
    total_sales = sum(s for _, s, _ in data)
    total_units = sum(u for _, _, u in data)

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["region", "sales", "units"])
    for row in data:
        writer.writerow(row)
    input_text = buf.getvalue()

    lines = ["Sales Report by Region", "======================", "Region    Sales    Units"]
    for reg, s, u in data:
        lines.append(f"{reg:<10}{s:<9}{u}")
    lines += ["", f"Total Sales: {total_sales}", f"Total Units: {total_units}"]
    expected_text = "\n".join(lines) + "\n"

    required = ["Sales Report"]
    for reg, s, _ in data:
        required.extend([reg, str(s)])
    required += [f"Total Sales: {total_sales}", f"Total Units: {total_units}"]

    meta = {
        "use_case": "uc4_csv_to_txt_report",
        "complexity": complexity,
        "input_format": "csv",
        "output_format": "txt",
        "description": "Regional sales CSV -> human-readable TXT report with totals",
        "prompt_hint": (
            "Produce a plain-text report titled 'Sales Report by Region', an "
            "aligned table of the regions, then two summary lines "
            "'Total Sales: <sum>' and 'Total Units: <sum>'."
        ),
        "content_accuracy_mode": "txt_substring",
        "required_substrings": required,
        "seed": seed,
    }
    return GeneratedCase(
        "uc4_csv_to_txt_report", complexity, "csv", "txt", input_text, expected_text, meta
    )
