"""Turn verified pair records into chat-format training examples + a disjoint split."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def to_chat_record(record: dict[str, Any]) -> dict[str, Any]:
    """Build the user/assistant chat turns the student is trained on.

    user  = the metadata envelope + the conversion instruction (what the model
            sees at inference — never the full file).
    assistant = the verified <analysis> + <script>.
    """
    env_json = json.dumps(record["envelope"], indent=2, default=str)
    user = (
        f"Metadata envelope:\n```json\n{env_json}\n```\n\n"
        f"Task: {record['instruction']}\n"
        f"Write a Python conversion script (reads sys.argv[1], writes sys.argv[2]; "
        f"stdlib + pandas only)."
    )
    assistant = (
        f"<analysis>{record['analysis']}</analysis>\n"
        f"<script>\n{record['script']}\n</script>"
    )
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def _bucket(case_id: str, seed: int) -> float:
    """Deterministic [0,1) hash of a case id (stable across runs/machines)."""
    h = hashlib.md5(f"{seed}:{case_id}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def split_records(
    records: list[dict[str, Any]],
    *,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Split records into train/val/test, disjoint by case_id, deterministically."""
    out: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for rec in records:
        b = _bucket(rec["case_id"], seed)
        if b < test_frac:
            out["test"].append(rec)
        elif b < test_frac + val_frac:
            out["val"].append(rec)
        else:
            out["train"].append(rec)
    return out
