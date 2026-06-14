# Frontier-model conversion cost probe

**What this measures:** the real OpenRouter cost + latency of having a frontier
model write a Python *conversion script* for one sample file per use-case
(UC1–UC5). Costs come from each response's `usage.cost` (OpenRouter's actual
billed amount, including input, reasoning, and output tokens); where absent they
are computed from list pricing. Scripts are not executed — this is cost-only.

Files: 15 successful calls across 3 model(s). Sample inputs are tiny (<600 B each) — treat all figures as a **lower bound**.

## Per-model totals

| Model | Calls | Total cost | Avg cost/file | Total latency | Avg latency | In tok | Out tok | Reason tok |
|-------|------:|-----------:|--------------:|--------------:|------------:|-------:|--------:|-----------:|
| Claude Opus 4.8 | 5 | $0.30493 | $0.06098 | 128.4s | 25.7s | 3565 | 11484 | 0 |
| GPT-5.5 | 5 | $0.39364 | $0.07873 | 209.2s | 41.8s | 2443 | 12714 | 5280 |
| DeepSeek V4 Pro | 5 | $0.01858 | $0.00372 | 105.8s | 21.2s | 2494 | 5932 | 1604 |

## Per-use-case cost (USD)

| UC | Claude Opus 4.8 | GPT-5.5 | DeepSeek V4 Pro |
|----|------:|------:|------:|
| uc1 | $0.06008 | $0.07545 | $0.00274 |
| uc2 | $0.06018 | $0.04529 | $0.00279 |
| uc3 | $0.04649 | $0.08854 | $0.00290 |
| uc4 | $0.07834 | $0.13563 | $0.00400 |
| uc5 | $0.05983 | $0.04872 | $0.00616 |

## Extrapolation (cost-only, one LLM call per file)

Assumes one independent script-generation call per file (the model re-reads
each file to catch its edge cases). Linear in file count; **input cost grows
further with real file size**, so production figures would be higher.

| Model | Cost / 1,000 files | Cost / 10,000 files |
|-------|-------------------:|--------------------:|
| Claude Opus 4.8 | $60.98 | $609.85 |
| GPT-5.5 | $78.73 | $787.27 |
| DeepSeek V4 Pro | $3.72 | $37.17 |

## Why this supports the data-morph thesis

Even on trivially small files, each conversion costs a metered API call with
latency measured in seconds. A distilled local Gemma student runs these same
conversions at zero marginal API cost — the gap widens linearly with volume
and with file size. See the per-10k-files column above.
