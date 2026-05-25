from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.format_pairs import split_records, to_chat_record  # noqa: E402


def _rec(case_id: str):
    return {
        "case_id": case_id,
        "envelope": {"format": "csv", "schema": {"row_count": 2}},
        "instruction": "Convert to nested JSON",
        "analysis": "group by user",
        "script": "import sys\nprint('x')",
    }


class TestToChatRecord:
    def test_shape_and_content(self):
        chat = to_chat_record(_rec("uc1/gen_000001"))
        assert list(chat.keys()) == ["messages"]
        roles = [m["role"] for m in chat["messages"]]
        assert roles == ["user", "assistant"]
        assert "Convert to nested JSON" in chat["messages"][0]["content"]
        assert "row_count" in chat["messages"][0]["content"]
        assert "<analysis>group by user</analysis>" in chat["messages"][1]["content"]
        assert "<script>" in chat["messages"][1]["content"]


class TestSplitRecords:
    def test_disjoint_and_deterministic(self):
        recs = [_rec(f"uc/gen_{i:06d}") for i in range(100)]
        a = split_records(recs, val_frac=0.1, test_frac=0.1, seed=0)
        b = split_records(recs, val_frac=0.1, test_frac=0.1, seed=0)
        # deterministic
        assert [r["case_id"] for r in a["train"]] == [r["case_id"] for r in b["train"]]
        # disjoint
        ids = {s: {r["case_id"] for r in a[s]} for s in ("train", "val", "test")}
        assert ids["train"].isdisjoint(ids["val"])
        assert ids["train"].isdisjoint(ids["test"])
        assert ids["val"].isdisjoint(ids["test"])
        # covers everything
        assert len(ids["train"]) + len(ids["val"]) + len(ids["test"]) == 100
        # roughly 80/10/10
        assert 5 <= len(ids["val"]) <= 15 and 5 <= len(ids["test"]) <= 15
