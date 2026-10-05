#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-.venv/bin/python}"

# Convert all datasets currently present under ./source_data into unified
# parquet shards under ./data. Supported formats:
#   - Monash / TSF zip
#   - M4 train/test CSV
#   - Chronos parquet (id/timestamp/target)
#   - LOTSA HuggingFace Arrow (item_id/target)
#
# Override examples:
#   DATASETS=Electricity,Solar bash convert_data.sh
#   MIN_SERIES_LENGTH=512 bash convert_data.sh
#   PYTHON=python3 bash convert_data.sh

INPUT_DIR="${INPUT_DIR:-./source_data}"
OUTPUT_DIR="${OUTPUT_DIR:-./data}"
DATASETS="${DATASETS:-all}"
SHARD_SIZE="${SHARD_SIZE:-5000}"
MIN_SERIES_LENGTH="${MIN_SERIES_LENGTH:-2048}"

"$PYTHON" scripts/convert_pretrain_data.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --datasets "$DATASETS" \
  --shard-size "$SHARD_SIZE" \
  --min-series-length "$MIN_SERIES_LENGTH"

echo "Conversion finished. Manifest: ${OUTPUT_DIR}/manifest.json"
