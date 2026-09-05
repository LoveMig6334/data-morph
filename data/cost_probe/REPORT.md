# Frontier-model conversion cost probe

**What this measures:** the real OpenRouter cost + latency of having a frontier
model write a Python *conversion script* for one sample file per use-case
(UC1–UC5). Scripts are not executed — this is cost-only.

**Cost source:** the **actual OpenRouter dashboard billing** (`BILLED_TOTAL`) is
the ground truth below. The per-response `usage.cost` field over-reported the true
spend by ~1.9x, so it is shown only for reference in the reconciliation table.

Files: 15 successful calls across 3 model(s). Sample inputs are tiny (<600 B each) — treat all figures as a **lower bound**.

## Per-model totals (actual billing)

| Model | Calls | Billed total | Billed/file | Total latency | Avg latency | In tok | Out tok | Reason tok |
|-------|------:|-------------:|------------:|--------------:|------------:|-------:|--------:|-----------:|
| Claude Opus 4.8 | 5 | $0.16700 | $0.03340 | 128.4s | 25.7s | 3565 | 11484 | 0 |
| GPT-5.5 | 5 | $0.20900 | $0.04180 | 209.2s | 41.8s | 2443 | 12714 | 5280 |
| DeepSeek V4 Pro | 5 | $0.00843 | $0.00169 | 105.8s | 21.2s | 2494 | 5932 | 1604 |

## Billing reconciliation (usage.cost vs dashboard)

| Model | Reported usage.cost | Actual billed | Ratio |
|-------|--------------------:|--------------:|------:|
| Claude Opus 4.8 | $0.30493 | $0.16700 | 1.83x |
| GPT-5.5 | $0.39364 | $0.20900 | 1.88x |
| DeepSeek V4 Pro | $0.01858 | $0.00843 | 2.20x |

## Per-use-case split (reported usage.cost, relative)

Per-UC dashboard figures aren't available (billing is per-model/day), so this
shows the *reported* usage.cost split — useful for the relative shape across
use-cases, not absolute dollars (see reconciliation above).

| UC | Claude Opus 4.8 | GPT-5.5 | DeepSeek V4 Pro |
|----|------:|------:|------:|
| uc1 | $0.06008 | $0.07545 | $0.00274 |
| uc2 | $0.06018 | $0.04529 | $0.00279 |
| uc3 | $0.04649 | $0.08854 | $0.00290 |
| uc4 | $0.07834 | $0.13563 | $0.00400 |
| uc5 | $0.05983 | $0.04872 | $0.00616 |

## Extrapolation (actual billing, one LLM call per file)

Billed/file = dashboard total / files. Assumes one independent script-generation
call per file (the model re-reads each file to catch its edge cases). Linear in
file count; **input cost grows further with real file size**, so production
figures would be higher.

| Model | Cost / 1,000 files | Cost / 10,000 files |
|-------|-------------------:|--------------------:|
| Claude Opus 4.8 | $33.40 | $334.00 |
| GPT-5.5 | $41.80 | $418.00 |
| DeepSeek V4 Pro | $1.69 | $16.86 |

## Why this supports the data-morph thesis

Even on trivially small files, each conversion costs a metered API call with
latency measured in seconds. A distilled local Gemma student runs these same
conversions at zero marginal API cost — the gap widens linearly with volume
and with file size. See the per-10k-files column above.
