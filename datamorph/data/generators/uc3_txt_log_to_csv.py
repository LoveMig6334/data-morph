"""Generator: bracketed-timestamp logs (TXT) -> CSV (timestamp,level,source,message)."""

from __future__ import annotations

import csv
import io
import random
from datetime import datetime, timedelta

from .base import GeneratedCase

_LEVELS = ["INFO", "WARN", "ERROR", "DEBUG"]
_SOURCES = ["app", "db", "auth", "cache", "api"]
# Comma-free messages keep the oracle CSV unquoted and exactly verifiable.
_MESSAGES = [
    "Server started",
    "Connection lost",
    "Slow query detected",
    "Reconnecting",
    "User login succeeded",
    "Cache miss",
    "Request handled",
    "Disk space low",
]
_N_LINES = {"simple": 6, "medium": 30, "complex": 120}


def generate(seed: int, complexity: str) -> GeneratedCase:
    rng = random.Random(seed)
    n = _N_LINES[complexity]
    t = datetime(2026, 4, 15, 10, 0, 0)

    in_lines: list[str] = []
    rows: list[tuple[str, str, str, str]] = []
    for _ in range(n):
        t = t + timedelta(seconds=rng.randint(1, 90))
        level = rng.choice(_LEVELS)
        source = rng.choice(_SOURCES)
        msg = rng.choice(_MESSAGES)
        in_lines.append(f"[{t:%Y-%m-%d %H:%M:%S}] {level} {source}: {msg}")
        rows.append((f"{t:%Y-%m-%dT%H:%M:%S}", level, source, msg))

    input_text = "\n".join(in_lines) + "\n"
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["timestamp", "level", "source", "message"])
    writer.writerows(rows)
    expected_text = buf.getvalue()

    meta = {
        "use_case": "uc3_txt_log_to_csv",
        "complexity": complexity,
        "input_format": "txt",
        "output_format": "csv",
        "description": "Bracketed-timestamp logs -> CSV timestamp,level,source,message",
        "prompt_hint": (
            "Each line: [YYYY-MM-DD HH:MM:SS] LEVEL source: message. Output CSV "
            "columns timestamp (ISO-8601 YYYY-MM-DDTHH:MM:SS), level, source, message."
        ),
        "seed": seed,
    }
    return GeneratedCase(
        "uc3_txt_log_to_csv", complexity, "txt", "csv", input_text, expected_text, meta
    )
