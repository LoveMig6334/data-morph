from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datamorph.features.format_pairs import split_records, to_chat_record  # noqa: E402


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


from scripts.build_dataset import build_dataset  # noqa: E402


class TestBuildDataset:
    def test_writes_three_jsonl_files(self, tmp_path):
        interim = tmp_path / "interim"
        interim.mkdir()
        for i in range(20):
            (interim / f"uc__gen_{i:06d}.json").write_text(
                __import__("json").dumps(_rec(f"uc/gen_{i:06d}")), encoding="utf-8"
            )
        # a non-record file that must be ignored
        (interim / "collect_manifest.json").write_text('{"n_cases": 20}', encoding="utf-8")

        processed = tmp_path / "processed"
        summary = build_dataset(interim, processed, seed=0)
        for split in ("train", "val", "test"):
            f = processed / f"{split}.jsonl"
            assert f.exists()
        # manifest was not counted as a record
        assert summary["n_records"] == 20
        # each line is a valid chat record
        import json as _json
        line = (processed / "train.jsonl").read_text().splitlines()[0]
        assert "messages" in _json.loads(line)
