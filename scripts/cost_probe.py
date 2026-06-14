"""Cost-probe harness.

Measures the real $ cost + latency of frontier models generating a file-conversion
*script* for one sample input per use-case (UC1-UC5). Cost-only: the generated
scripts are NOT executed. Produces evidence for the problem statement that relying
directly on a frontier LLM for file conversion is expensive at scale.

Usage:
    python scripts/cost_probe.py            # run all models x all files
    python scripts/cost_probe.py --dry-run  # print what would be sent, no API calls

Reads OPENROUTER_API_KEY from .env. Writes raw responses to data/cost_probe/raw/,
aggregate results to data/cost_probe/results.json, and a human report to
data/cost_probe/REPORT.md.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "data" / "cost_probe"
INPUTS = PROBE / "inputs"
RAW = PROBE / "raw"
SKILL = ROOT / "skills" / "cost_probe_conversion.md"

API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Confirmed slugs + pricing (USD per token) pulled from GET /api/v1/models on
# 2026-06-14. Used as a fallback if a response omits usage.cost.
MODELS = [
    {"label": "Claude Opus 4.8", "slug": "anthropic/claude-opus-4.8", "in": 0.000005,   "out": 0.000025},
    {"label": "GPT-5.5",         "slug": "openai/gpt-5.5",            "in": 0.000005,   "out": 0.00003},
    {"label": "DeepSeek V4 Pro", "slug": "deepseek/deepseek-v4-pro",  "in": 0.000000435,"out": 0.00000087},
]


def load_api_key() -> str:
    env = ROOT / ".env"
    for line in env.read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("OPENROUTER_API_KEY not found in .env")


def build_prompt(template: str, entry: dict, content: str) -> str:
    return (
        template.replace("{src}", entry["src_format"])
        .replace("{tgt}", entry["tgt_format"])
        .replace("{task}", entry["task"])
        .replace("{content}", content)
    )


def call_model(slug: str, prompt: str, key: str) -> tuple[dict, float]:
    """POST one chat completion. Returns (parsed_json, latency_seconds)."""
    body = json.dumps(
        {
            "model": slug,
            "messages": [{"role": "user", "content": prompt}],
            "usage": {"include": True},
        }
    ).encode()
    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    return data, time.monotonic() - start


def extract_usage(resp: dict, model: dict) -> dict:
    """Pull token counts + cost from a response; compute cost if not provided."""
    u = resp.get("usage", {}) or {}
    prompt_tok = u.get("prompt_tokens", 0)
    completion_tok = u.get("completion_tokens", 0)
    # reasoning tokens are usually counted inside completion_tokens by OpenRouter
    details = u.get("completion_tokens_details", {}) or {}
    reasoning_tok = details.get("reasoning_tokens", 0)
    cost = u.get("cost")
    cost_source = "openrouter"
    if cost is None:
        cost = prompt_tok * model["in"] + completion_tok * model["out"]
        cost_source = "computed"
    return {
        "prompt_tokens": prompt_tok,
        "completion_tokens": completion_tok,
        "reasoning_tokens": reasoning_tok,
        "cost_usd": cost,
        "cost_source": cost_source,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="don't call the API")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    template = SKILL.read_text()
    manifest = json.loads((INPUTS / "manifest.json").read_text())
    key = None if args.dry_run else load_api_key()

    records = []
    for uc, entry in manifest.items():
        content = (INPUTS / entry["file"]).read_text()
        prompt = build_prompt(template, entry, content)
        for model in MODELS:
            tag = f"{model['slug'].replace('/', '__')}__{uc}"
            print(f"[{model['label']:16}] {uc} ({entry['src_format']}->{entry['tgt_format']}) ...", flush=True)
            if args.dry_run:
                print(f"  prompt chars: {len(prompt)}")
                continue
            try:
                resp, latency = call_model(model["slug"], prompt, key)
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")[:300]
                print(f"  SKIP {e.code}: {msg}")
                records.append({"uc": uc, "model": model["label"], "slug": model["slug"], "error": f"{e.code}: {msg}"})
                continue
            except Exception as e:  # noqa: BLE001
                print(f"  SKIP error: {e}")
                records.append({"uc": uc, "model": model["label"], "slug": model["slug"], "error": str(e)})
                continue
            (RAW / f"{tag}.json").write_text(json.dumps(resp, indent=2))
            usage = extract_usage(resp, model)
            rec = {"uc": uc, "model": model["label"], "slug": model["slug"],
                   "src": entry["src_format"], "tgt": entry["tgt_format"],
                   "latency_s": round(latency, 3), **usage}
            records.append(rec)
            print(f"  cost=${usage['cost_usd']:.6f} ({usage['cost_source']}) "
                  f"in={usage['prompt_tokens']} out={usage['completion_tokens']} "
                  f"reason={usage['reasoning_tokens']} {latency:.1f}s")

    if args.dry_run:
        return

    (PROBE / "results.json").write_text(json.dumps(records, indent=2))
    write_report(records)
    print(f"\nWrote {PROBE/'results.json'} and {PROBE/'REPORT.md'}")


def write_report(records: list[dict]) -> None:
    ok = [r for r in records if "error" not in r]
    errs = [r for r in records if "error" in r]
    by_model: dict[str, list[dict]] = {}
    for r in ok:
        by_model.setdefault(r["model"], []).append(r)

    lines = [
        "# Frontier-model conversion cost probe",
        "",
        "**What this measures:** the real OpenRouter cost + latency of having a frontier",
        "model write a Python *conversion script* for one sample file per use-case",
        "(UC1–UC5). Costs come from each response's `usage.cost` (OpenRouter's actual",
        "billed amount, including input, reasoning, and output tokens); where absent they",
        "are computed from list pricing. Scripts are not executed — this is cost-only.",
        "",
        f"Files: {len(ok)} successful calls across {len(by_model)} model(s). "
        f"Sample inputs are tiny (<600 B each) — treat all figures as a **lower bound**.",
        "",
        "## Per-model totals",
        "",
        "| Model | Calls | Total cost | Avg cost/file | Total latency | Avg latency | In tok | Out tok | Reason tok |",
        "|-------|------:|-----------:|--------------:|--------------:|------------:|-------:|--------:|-----------:|",
    ]
    summary = {}
    for model, rows in by_model.items():
        n = len(rows)
        tcost = sum(r["cost_usd"] for r in rows)
        tlat = sum(r["latency_s"] for r in rows)
        tin = sum(r["prompt_tokens"] for r in rows)
        tout = sum(r["completion_tokens"] for r in rows)
        treason = sum(r["reasoning_tokens"] for r in rows)
        summary[model] = {"avg_cost": tcost / n, "avg_lat": tlat / n}
        lines.append(
            f"| {model} | {n} | ${tcost:.5f} | ${tcost/n:.5f} | {tlat:.1f}s | "
            f"{tlat/n:.1f}s | {tin} | {tout} | {treason} |"
        )

    lines += ["", "## Per-use-case cost (USD)", "",
              "| UC | " + " | ".join(by_model.keys()) + " |",
              "|----|" + "|".join(["------:"] * len(by_model)) + "|"]
    ucs = sorted({r["uc"] for r in ok})
    for uc in ucs:
        cells = []
        for model in by_model:
            match = next((r for r in by_model[model] if r["uc"] == uc), None)
            cells.append(f"${match['cost_usd']:.5f}" if match else "—")
        lines.append(f"| {uc} | " + " | ".join(cells) + " |")

    lines += ["", "## Extrapolation (cost-only, one LLM call per file)", "",
              "Assumes one independent script-generation call per file (the model re-reads",
              "each file to catch its edge cases). Linear in file count; **input cost grows",
              "further with real file size**, so production figures would be higher.", "",
              "| Model | Cost / 1,000 files | Cost / 10,000 files |",
              "|-------|-------------------:|--------------------:|"]
    for model, s in summary.items():
        lines.append(f"| {model} | ${s['avg_cost']*1000:,.2f} | ${s['avg_cost']*10000:,.2f} |")

    if errs:
        lines += ["", "## Skipped / errors", ""]
        for e in errs:
            lines.append(f"- **{e['model']}** ({e['slug']}) {e.get('uc','')}: {e['error']}")

    lines += ["", "## Why this supports the data-morph thesis", "",
              "Even on trivially small files, each conversion costs a metered API call with",
              "latency measured in seconds. A distilled local Gemma student runs these same",
              "conversions at zero marginal API cost — the gap widens linearly with volume",
              "and with file size. See the per-10k-files column above.", ""]

    (PROBE / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
