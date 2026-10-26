"""Filesystem locations used by the experiments.

The original `decovaefm/` package lives in the repository root, two levels above this folder.
Data and run outputs can be moved with environment variables:
  DECOVAE_VER0  pretraining corpus produced by convert_data.sh (default: <repo>/data)
  DECOVAE_RUNS  checkpoints, caches and per-run tables (default: experiments/decomposition/runs, git-ignored)
"""
import os
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parents[1]
PERIOD_DECOMP_DIR = EXPERIMENTS_DIR / 'period_decomp'
BENCHMARK_DIR = EXPERIMENTS_DIR / 'benchmark'
VER0_DIR = Path(os.environ.get('DECOVAE_VER0', REPO_ROOT / 'data'))
RUNS_DIR = Path(os.environ.get('DECOVAE_RUNS', EXPERIMENTS_DIR / 'runs'))
