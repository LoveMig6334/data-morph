# Project Progression

Living tracker for **data morph** (AI Builders 2026). Update this file at the
end of every working session — keep it short, keep it honest.

- **Today:** 2026-05-06
- **Plan week (calendar):** W5 — Fine-tune Gemma 2B (LoRA)
- **Actual progress:** end of W2 — baseline shipped, W3 not started
- **Status:** behind plan by ~2 weeks

---

## Weekly status

| Week | Plan focus | State | Evidence |
|------|------------|-------|----------|
| W1 | Problem + 5 use cases | ✅ Done | `README.md`, `CLAUDE.md` |
| W2 | Metrics + Opus baseline | ✅ Done | `src/evaluation/`, `tests/test_metrics.py` (28 tests), `docs/W2_baseline.md`, `results/baseline_2026-04-17_192259/` |
| W3 | 500–1000 teacher pairs | ⬜ Not started | `src/data/`, `src/features/` empty |
| W4 | EDA + cleaning | ⬜ Not started | `notebooks/` empty |
| W5 | Gemma 2B + LoRA | ⬜ Not started (needs Mac/Colab) | `src/models/` empty |
| W6 | Eval + error analysis | ⬜ Not started | — |
| W7 | pip package + HF Hub | ⬜ Not started | — |
| W8 | Blog / slides / poster | ⬜ Not started | — |

Grading target: ≥70 / 100. Points landed so far: **~30** (W1 + W2).

---

## W2 baseline — headline numbers

15 hand-crafted cases, 5 use cases, Claude Opus teacher, 0 teacher errors.

| Metric | Score | Student target (≥80%) |
|--------|------:|----------------------:|
| Format Validity | 1.000 | ≥ 0.800 |
| Schema Compliance | 0.867 | ≥ 0.694 |
| Loadability | 1.000 | ≥ 0.800 |
| Content Accuracy | 0.983 | ≥ 0.786 |

Only weak spot: **UC1 csv → nested JSON** (`SC 0.333`) — Opus emits CSV
numerics as JSON strings and drops the entity wrapper when collapsing
multi-row entities. Both are concrete capabilities the student must learn.

---

## Open issues / risks

- **Behind schedule** — W3 + W4 still owed. Cutting advanced use cases
  is preferable to cutting deployment per the plan's risk register.
- **MLX/Unsloth wheel verification (W5)** — Project standardized on
  **Python 3.12** (decision: MLX support is stronger on 3.12). Before W5
  starts, confirm Unsloth + MLX-LM wheels actually install on macOS
  Python 3.12 — if not, the documented fallback is Colab + PyTorch + Unsloth.
- **Test set is 15 cases** — plan calls for 50–100. Expand during W3.
- **No Agent Skill in `claude -p`** — skill is referenced as a file. A future
  Anthropic Agent SDK pipeline would likely raise scores 5–15%.
- **Single teacher run per case** — no sampling variance measured. W3 data
  generation should add a retry + verification pass per pair.
- **Mac handoff** — M5 Max scheduled for 2026-04-30. If unavailable, the
  documented fallback is **Colab + PyTorch + Unsloth**, not local Windows.

---

## Next up

1. Start W3: design the teacher pair-generation loop in `src/data/`,
   write a verifier that gates every pair before it lands in
   `data/processed/`. Aim for ~50 pairs as a first batch, then scale.
2. Expand the test set toward 50 cases (disjoint from training) and
   re-run the baseline to confirm scores hold on the larger set.
3. Before W5, smoke-test `mlx-lm` + `unsloth` install + import on
   the 3.12 venv. If either fails, switch to Colab fallback early.

---

## Update log

Append one bullet per session. Newest on top. Keep each entry under 2 lines.

- **2026-05-06** — Standardized on **Python 3.12** (MLX support is
  stronger). Updated `CLAUDE.md` env section + handoff note and
  `docs/W2_baseline.md` accordingly; switched install instructions
  from `pip install -r requirements.txt` to `uv sync` / `uv add`.
- **2026-05-06** — Declared real deps in `pyproject.toml` (`pandas`
  runtime, `pytest` dev) via `uv add`; fixed placeholder description.
  All 28 metric tests pass on the 3.12 venv. Added a tracking rule to
  `CLAUDE.md` requiring this log to be updated on every finished item.
- **2026-05-06** — Created this progression tracker. Confirmed repo state:
  W1+W2 done, W3+ not started; flagged Python version drift in `pyproject.toml`.
