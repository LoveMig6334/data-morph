# data morph

Machine learning project.

## Setup

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Structure

```
data/
  raw/          # original, immutable data
  interim/      # intermediate transformations
  processed/    # final modeling-ready data
notebooks/      # exploration & analysis
src/
  data/         # data loading / ingestion
  features/     # feature engineering
  models/       # training / inference
tests/          # unit tests
models/         # saved model artifacts
```
