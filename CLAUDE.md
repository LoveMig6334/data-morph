# CLAUDE.md

Guidance for Claude Code when working in this repo.

## Project snapshot

**data morph** — AI Builders 2026 project. Distill a file-format-conversion capability from Claude Opus (teacher, via Agent Skill) into a fine-tuned Gemma 2B (student, via LoRA/QLoRA). Supported formats: CSV, JSON, TXT.

Ship target: `pip install`-able library + model on Hugging Face Hub.

## Platform reality check

The plan assumes **MacBook Pro M5 Max + MLX**. The user's Mac (custom spec) arrives **2026-04-30**; until then the dev machine is **Windows 11**.

**Machine-handoff schedule:**

| Week | Task | Machine |
|------|------|---------|
| W1 | Problem + use cases | Windows |
| W2 | Metrics + Claude Opus baseline | Windows |
| W3 | Teacher-generated training pairs (Claude API) | Windows or Mac |
| W4 | EDA | Either |
| W5 | LoRA fine-tune Gemma 2B — **requires MLX** | Mac only |
| W6 | Eval + error analysis | Mac |
| W7 | pip package + HF Hub | Mac |
| W8 | Blog / slides / poster | Mac |

When touching training or inference code:
- Don't hardcode `mlx` imports or Apple-only paths at the top level.
- Isolate framework code behind an abstraction (`src/models/backend.py` style) with an MLX implementation and a PyTorch + Unsloth fallback for Colab/Windows.
- If asked to run actual fine-tuning on this Windows box before 2026-04-30, stop and confirm — the plan's fallback is **Google Colab + PyTorch + Unsloth**, not local Windows training.
- When the Mac arrives: make a **fresh** `.venv` on macOS (same Python 3.12) and re-install from the lockfile (`uv sync`). Don't try to sync venvs across OSes.

## Environment

- **Python 3.12** (venv at `.venv/`, managed by `uv`). We landed on 3.12 because **MLX support is stronger on 3.12 than 3.11**. Don't regenerate with a different version without asking.
- Activate: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (mac).
- Install / sync: `uv sync` (resolves from `pyproject.toml` + `uv.lock`). Add new deps with `uv add <pkg>` (or `uv add --dev <pkg>` for dev-only).

## Folder conventions

```
data/raw/        immutable source files — never modify in place
data/interim/    teacher outputs before verification
data/processed/  only verified pairs enter here → used for training
notebooks/       exploration; graduate reusable code into src/
src/data/        collection + teacher pair generation
src/features/    formatting into (instruction, input, output)
src/models/      fine-tune + inference
tests/           unit tests + the automated eval pipeline
models/          checkpoints (gitignored)
```

Rule: if a notebook cell is run more than twice in two different notebooks, move it into `src/`.

## The four metrics (do not invent new ones without asking)

1. **Format Validity** — output loads with `json.load()` / `csv.reader()` / etc.
2. **Schema Compliance** — JSON Schema / expected structure passes.
3. **Loadability** — pandas / downstream lib can consume it.
4. **Content Accuracy** — field-level comparison vs source, no hallucination, no data loss.

The evaluation script in `tests/` (or `src/` — TBD) must compute all four on every model output.

## Data rules

- Target: 500–1000 teacher-generated training pairs. Every pair is verified programmatically (`json.load`, `csv.reader`, schema check) **before** it enters `data/processed/`.
- Test set: 50–100 files, disjoint from training, tagged `simple | medium | complex`.
- Never commit raw data or model checkpoints — `.gitignore` already excludes them.

## Model rules

- Student: **Gemma 2B**, fine-tuned with **LoRA / QLoRA** (parameter-efficient only — no full fine-tune).
- Teacher: **Claude Opus + Claude Code + Agent Skill** for pair generation.
- Target: student achieves **≥80% of teacher accuracy** on every metric.
- Fallback if 2B underperforms: Gemma 7B (flagged in the risk register).

## Risks to keep in mind (from plan §7)

- Teacher output quality → always verify pairs.
- Gemma 2B capacity → start from simple use cases, escalate to 7B only if needed.
- MLX incompatibility → Colab + PyTorch + Unsloth is the documented fallback.
- 8-week timeline → prefer cutting advanced use cases over cutting deployment.

## Collaboration notes for Claude

- This is a student project for AI Builders 2026 — grading matters. When in doubt, map work back to the grading criteria table in the README / plan.
- The user is building from scratch on Windows but planning to train on macOS/Colab. Don't assume any CUDA, MLX, or GPU is available on the local machine unless the user confirms.
- Prefer small, reviewable changes. The user is actively learning — explain non-obvious ML/Python choices briefly.

## Tracking progress

- **`docs/progression.md` is the living progress tracker.** Every time you finish a meaningful piece of work — a feature, a fix, a config change, a milestone — append a dated bullet to its **Update log** section (newest on top, ≤ 2 lines per entry).
- If the work changes weekly status, scores, open issues, or "next up", update those sections too. Keep the doc concise — it's a tracker, not a journal.
- "Done with something" includes: code merged, dependencies added, eval runs completed, plan-week finished. It does *not* include exploratory reads or aborted attempts.
