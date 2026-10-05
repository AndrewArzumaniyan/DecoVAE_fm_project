#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-.venv/bin/python}"

# Staged hyperparameter search on electricity/pedestrian_counts/solar.
# Thinning via --window-stride (not load-fraction; few parquet shards).
# Best config selected by validation WQL @ quantiles 0.1..0.9
#
# Stages:
#   1) dynamics:  lr × kl_weight
#   2) capacity:  d_model × enc_layers × n_heads
#   3) patching:  (patch:stride) × k_top
#
# Outputs: ./outputs/search/{results.csv,best_params.json,best.ckpt,train_full.sh}
# After search, train with: python scripts/train.py experiment=base

"$PYTHON" scripts/search_hyperparams.py \
  --datasets electricity,pedestrian_counts,solar \
  --accelerator gpu \
  --devices 1 \
  --batch-size 256 \
  --num-workers 8 \
  --window-stride 384 \
  --epochs 8 \
  --stages dynamics,capacity,patching \
  --learning-rates 3e-5,1e-4,3e-4 \
  --kl-weights 1e-4,1e-3,1e-2 \
  --d-models 64,128 \
  --enc-layers-grid 2,3 \
  --dec-layers-grid 2 \
  --n-heads-grid 4,8 \
  --patch-stride-pairs 16:8,16:16,32:16 \
  --k-tops 8,16,32 \
  --search-dir ./outputs/search \
  --full-epochs 50 \
  --full-window-stride 96

echo "Search finished. Best params: ./outputs/search/best_params.json"
echo "Run full training with: python scripts/train.py experiment=base"
