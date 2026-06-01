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

**W1–W6 complete; W7 model surgery done — a 2.0 GB single-file student is production-validated.**

- **Data (W3):** 800 verified teacher pairs (100% accept), split into
  `data/processed/{train,val,test}.jsonl` (650 / 80 / 70, content-disjoint).
- **EDA (W4):** `notebook/w4_eda.ipynb` — training-readiness audit (balance,
  leakage, sequence-length budget).
- **Fine-tune (W5):** Gemma-4 E2B distilled via LoRA (`mlx_vlm.lora`, SFT) on the
  envelope→script task. Best checkpoint (iter-400) selected by held-out eval.
- **Eval (W6):** on the held-out 70-case test set, through the full pipeline
  (envelope → script → sandbox → 4 metrics), the fine-tuned student reaches
  **65/70 one-shot** and **68/70 (0.971) at production retry≤3** — already ≥80%-of-teacher.
- **Shrink (W7):** the multimodal base is mostly dead weight for this task. A
  three-step surgery (`scripts/build_textonly_student.py` + `prune_vocab.py`) fuses the
  adapter, strips the unused **vision + audio towers**, prunes the **262 k vocab → 16 k**
  (the corpus uses ~4.5 k tokens; the vocab indexes the two biggest tensors), then
  re-quantizes — all on a pure `gemma4_text` model loaded via `mlx_lm`:

  | Artifact | params | size | retry≤3 | % teacher |
  |---|---:|---:|---:|---:|
  | fine-tuned bf16 (runtime adapter) | 5.12 B | 9.6 GB | — | — |
  | *prior 8-bit (full model)* | 5.1 B | 5.5 GB | 68/70 | ~97% |
  | fused + text-only + vocab-16k, bf16 | 2.05 B | 3.8 GB | **69/70 (0.986)** | ~99% |
  | **+ 8-bit (final ship artifact)** | **2.05 B** | **2.0 GB** | **67/70 (0.957)** | **~96%** |

  **9.6 GB → 2.0 GB (−79%)** with accuracy still well above the **≥80%-of-teacher**
  target on every metric. Each cut is lossless-by-construction (strip/prune, guarded by
  a tokenizer round-trip verification gate) or a small retry-recoverable numerical cost.

**Next (W7 deployment):** push the 2.0 GB model to Hugging Face Hub with a model card,
ship the `pip`-installable pipeline wrapper. See `docs/progression.md` for the live tracker.

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
notebook/       # EDA (w4_eda), fine-tune scaffold (w5_finetune), experiments
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
