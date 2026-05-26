#!/usr/bin/env bash
#
# Parallel teacher-pair collection, sharded by use case.
#
# Runs one `collect_pairs.py` process per use case concurrently (each pointed at
# a single-use-case root via a symlink), all writing unique record files into
# data/interim. Cuts the full 800-case run from ~8h sequential to ~1.5h.
#
# Each shard's progress: data/interim/_<use_case>.log
# Note: the five shards each overwrite collect_manifest.json (last-writer-wins),
# so trust the "Accepted records" count below — the pairs themselves are safe
# (every accepted case writes a uniquely-named <use_case>__<case>.json).
#
# Usage:  bash scripts/collect_all_parallel.sh
#
set -uo pipefail

cd "$(dirname "$0")/.."
RAW="data/raw"
INTERIM="data/interim"
SHARD_ROOT="${TMPDIR:-/tmp}/morph_shard"

if [ ! -d "$RAW" ] || [ -z "$(ls -d "$RAW"/*/ 2>/dev/null)" ]; then
  echo "No corpus found in $RAW. Generate it first:"
  echo "  uv run python scripts/generate_corpus.py --count 800 --dest data/raw"
  exit 1
fi

echo "Clearing previous interim records..."
rm -f "$INTERIM"/*.json "$INTERIM"/_*.log
rm -rf "$SHARD_ROOT"
mkdir -p "$INTERIM"

pids=()
for dir in "$RAW"/*/; do
  uc=$(basename "$dir")
  root="$SHARD_ROOT/$uc"
  mkdir -p "$root"
  ln -sfn "$(pwd)/$RAW/$uc" "$root/$uc"
  echo "  launching shard: $uc"
  uv run python scripts/collect_pairs.py --raw "$root" --interim "$INTERIM" \
      > "$INTERIM/_$uc.log" 2>&1 &
  pids+=($!)
done

echo "Waiting for ${#pids[@]} shards to finish (tail data/interim/_<use_case>.log to watch)..."
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done

echo ""
echo "=== shard summaries ==="
for log in "$INTERIM"/_*.log; do
  printf '%-32s %s\n' "$(basename "$log")" "$(tail -n 1 "$log")"
done

accepted=$(find "$INTERIM" -maxdepth 1 -name '*.json' ! -name 'collect_manifest.json' | wc -l | tr -d ' ')
echo ""
echo "Accepted records in $INTERIM: $accepted"
[ "$fail" -eq 0 ] || echo "WARNING: a shard exited non-zero — inspect the logs above."

echo ""
echo "Building train/val/test dataset..."
uv run python scripts/build_dataset.py --interim "$INTERIM" --processed data/processed
echo ""
echo "Done. Training files: data/processed/{train,val,test}.jsonl"
