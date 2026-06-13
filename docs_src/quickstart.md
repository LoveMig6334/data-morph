# Quickstart

## Install

data morph runs on **Apple Silicon** via MLX. Install the package with the `mlx` extra:

```bash
pip install "data-morph-gemma[mlx]"
```

!!! note "Import vs. install name"
    The distribution is **`data-morph-gemma`** (`pip install`), but the import name is
    **`datamorph`**. PyPI blocks `data-morph` as too similar to an existing project.

The 2.0 GB model **downloads automatically** from the Hugging Face Hub on first use and is
cached under `~/.cache/huggingface`. To use a local copy instead, set `GEMMA_MLX_MODEL` to
its path.

## Convert a file (Python)

```python
from datamorph import convert_file

result = convert_file("contacts.csv", "contacts.json")

print(result.accepted)        # True if the output passed validation
print(result.scores)          # {'format_validity': 1.0, 'loadability': 1.0}
print(result.output_path)     # contacts.json (written when an output path is given)
```

`convert_file` runs the full pipeline — extract a metadata envelope, have the model write
a conversion script, run it in a sandbox, and validate the result — retrying up to three
times on failure. See [`ConversionResult`](api.md) for every field.

Formats are auto-detected from file extensions; override them when needed:

```python
convert_file("logs.txt", input_format="txt", output_format="csv")  # prints to stdout if no output path
```

## Convert a file (command line)

```bash
datamorph convert contacts.csv contacts.json
# datamorph: csv -> json ok (retries=0, scores={...}) -> contacts.json
```

```bash
datamorph convert app.log --output-format csv > events.csv   # pipe to stdout
datamorph --version
```

Exit codes: **0** = converted and validated, **1** = ran but failed validation, **2** =
usage / input error.

## Supported conversions

CSV, JSON, and TXT, in five patterns: CSV→JSON (nested), JSON→CSV (flatten), TXT log→CSV,
CSV→TXT (report), and schema migration.
