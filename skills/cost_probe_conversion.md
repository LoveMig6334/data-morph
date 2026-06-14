---
name: cost_probe_conversion
description: Shared prompt used by the cost-probe harness. Each frontier model receives this same instruction (with {src}, {tgt}, {task}, {content} filled in) and must write a Python conversion script. Used to measure the real $ cost of frontier-model file conversion at proof-of-concept scale.
---

# File-conversion script generator (cost-probe prompt)

You are an expert data engineer. You are given the **full contents** of a source
`{src}` file. Your job is to write a single self-contained Python script that
converts it to `{tgt}`.

Task for this file: {task}

Requirements for the script:
- Read the source from a path given as `sys.argv[1]` and write the converted
  result to `sys.argv[2]`.
- Use only the Python 3.12 standard library (`csv`, `json`, `re`, etc.).
- Handle realistic edge cases for this conversion (missing/empty fields,
  quoting/escaping, type coercion, nested structures, malformed lines) so the
  same script would work on other files of the same kind — not just this sample.
- Produce valid, well-formed `{tgt}` output.

Respond with your reasoning, then the script inside a single fenced code block:

```python
# your script here
```

Here is the full content of the source `{src}` file:

----- BEGIN FILE -----
{content}
----- END FILE -----
