# data morph

**Open Source File Data Migration with Fine-tuned Small Language Model**

Knowledge distillation from a large-model agent (Claude Opus + Agent Skill) into a fine-tuned Gemma 2B, so developers can convert between file formats locally for free instead of paying for frontier-LLM API calls.

AI Builders 2026 · Track: Agentic AI + NLP

## Problem

Rule-based parsers can't handle messy, context-dependent file conversions. Frontier LLMs can, but they're expensive at scale. This project distills that capability into a 2B-parameter model that runs locally.

## Approach

1. **Teacher**: Claude Opus + Claude Code + Agent Skill generates 500–1000 verified training pairs.
2. **Student**: Gemma 2B, fine-tuned with LoRA / QLoRA.
3. **Target**: ≥80% of teacher accuracy across 4 metrics — Format Validity, Schema Compliance, Loadability, Content Accuracy.

### Pipeline architecture

Conversion is a **five-stage pipeline**, not a single end-to-end model call.
The model only ever sees a small structured metadata envelope, never the
full source file:

```
[source file]
    │
    ├─→ [1. Metadata extractor]  deterministic — schema + samples + warnings
    ├─→ [2. Context summarizer]  Gemma 2B base — short NL summary
    ↓
[3. Script generator]   Claude Opus (training) → Gemma 2B fine-tuned (inference)
    ↓ outputs an executable Python script
[4. Sandbox executor]   deterministic — runs the script
    ↓ converted output file
[5. Validator]          the 4 W2 metrics — format, schema, load, content
    ↓
[output file]
```

**Why this shape**: distillation target narrows from "transform a whole
file" (impractical for a 2 B model) to "read metadata, write a script"
(realistic). The model never sees full file content, so the pipeline scales
to arbitrary file sizes. Failures are debuggable — the script is a readable
intermediate artefact.

### Status

W1–W2 complete (metrics + Opus baseline). Stage 1 extractors (CSV, JSON, TXT),
the Stage 4 sandbox, and the full Stage 3 teacher pipeline are **built and
validated end-to-end** — a 10-case stratified dry-run accepts 100% across all
five use cases. Source data comes from **seeded synthetic generators** (an
800-case corpus, reproducible via `scripts/generate_corpus.py`) that act as the
ground-truth oracle. Next: the full teacher-collection run → training dataset →
W5 LoRA fine-tune. See `docs/progression.md` for the live tracker.

## Supported formats

CSV, JSON, TXT — in 5 use cases (CSV→JSON nested, JSON→CSV flattening, TXT log→CSV, CSV→TXT report, schema migration).

## Setup

Requires **Python 3.12** (chosen for stronger MLX support). Project is
managed by [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                        # creates .venv from pyproject.toml + uv.lock
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

Add a new dependency: `uv add <pkg>` (or `uv add --dev <pkg>` for dev-only).

## Hardware / framework

- **Primary target**: MacBook Pro M5 Max (40 GPU cores, 120 GB unified memory) with **MLX**.
- **Fallback**: Google Colab + PyTorch + Unsloth (used when MLX is unavailable, e.g. on Windows).

## Repo structure

```
data/
  raw/          # synthetic corpus from seeded generators (regenerable, gitignored)
  interim/      # verified teacher pairs (envelope + analysis + script + scores)
  processed/    # train/val/test chat JSONL for fine-tuning
  test_set/     # 15 hand-crafted W2 baseline cases
notebooks/      # EDA, error analysis, experiments
src/
  extractor/    # Stage 1: deterministic metadata extractor — CSV, JSON, TXT (done)
  evaluation/   # Stage 5: the 4 W2 metrics + Opus-baseline runner (DO NOT EDIT)
  data/         # generators (oracle), sandbox (Stage 4), teacher_script + collect (Stage 3)
  features/     # format_pairs: verified pairs → chat JSONL + disjoint split
  models/       # LoRA/QLoRA fine-tune + inference (W5)
scripts/        # generate_corpus, collect_pairs, collect_all_parallel, build_dataset, baseline, plotting
skills/         # Agent-Skill prompts read by `claude -p` (file conversion + script generation)
tests/          # unit tests (metrics, extractor, data, features) + fixtures
models/         # Gemma-4 E2B (local, gitignored) + fine-tuned checkpoints
results/        # baseline run artefacts (per-run summary.json + plots)
docs/           # specs, plans, weekly reports (gitignored)
```

## Timeline (8 weeks)

| Week | Focus | Points |
|------|-------|-------:|
| 1 | Problem statement + use cases | 15 |
| 2 | Metrics + Claude Opus baseline | 15 |
| 3 | Teacher-generated training pairs | 15 |
| 4 | EDA + data cleaning | 20 |
| 5 | Fine-tune Gemma 2B (LoRA) | — |
| 6 | Evaluation + error analysis | 20 |
| 7 | Deployment (pip + HF Hub) | 15 |
| 8 | Blog, slides, poster | — |
| | **Total** | **100** (≥70 to pass) |

## Deliverables

- GitHub repo (this one)
- Hugging Face Hub model + model card
- `pip install`-able Python package
- Medium blog post
- Presentation slides + A1 poster
- Facebook post (100–200 words)

## Ethics

- Converted files may contain personal data → no uploads of user input.
- Teacher bias propagates to student — documented in model card.
- Hallucination risk mitigated by automated format/schema validation at inference time.
