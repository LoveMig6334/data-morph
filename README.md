# data morph

**Open Source File Data Migration with Fine-tuned Small Language Model**

Knowledge distillation from a large-model agent (Claude Opus + Agent Skill) into a fine-tuned Gemma 2B, so developers can convert between file formats locally for free instead of paying for frontier-LLM API calls.

AI Builders 2026 · Track: Agentic AI + NLP

## Problem

Rule-based parsers can't handle messy, context-dependent file conversions. Frontier LLMs can, but they're expensive at scale. This project distills that capability into a 2B-parameter model that runs locally.

## Approach

1. **Teacher**: Claude Opus + Claude Code + Agent Skill generates 500–1000 verified `(instruction, input, output)` training pairs.
2. **Student**: Gemma 2B, fine-tuned with LoRA / QLoRA.
3. **Target**: ≥80% of teacher accuracy across 4 metrics — Format Validity, Schema Compliance, Loadability, Content Accuracy.

## Supported formats

CSV, JSON, TXT — in 5 use cases (CSV→JSON nested, JSON→CSV flattening, TXT log→CSV, CSV→TXT report, schema migration).

## Setup

Requires **Python 3.11+**.

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # macOS/Linux
pip install -r requirements.txt
```

## Hardware / framework

- **Primary target**: MacBook Pro M5 Max (40 GPU cores, 120 GB unified memory) with **MLX**.
- **Fallback**: Google Colab + PyTorch + Unsloth (used when MLX is unavailable, e.g. on Windows).

## Repo structure

```
data/
  raw/          # source files collected from Kaggle / HF / GitHub (gitignored)
  interim/      # teacher-generated pairs pre-verification
  processed/    # verified training set for fine-tuning
notebooks/      # EDA, error analysis, experiments
src/
  data/         # data collection + teacher-model pair generation
  features/     # formatting into (instruction, input, output)
  models/       # LoRA/QLoRA fine-tune + inference
tests/          # unit tests + evaluation pipeline
models/         # fine-tuned checkpoints (gitignored)
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
