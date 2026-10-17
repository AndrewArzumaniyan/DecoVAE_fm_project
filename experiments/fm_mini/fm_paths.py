"""Filesystem locations of the mini-FM experiments.

The mini-FM code reuses the benchmark modules in ../decomposition/benchmark (data, runner, adapters, fourier_arm, metrics)
and the period estimator in ../decomposition/period_decomp; importing this module puts them on sys.path.
Locations can be moved with environment variables:
  DECOVAE_VER0  pretraining corpus produced by convert_data.sh (default: <repo>/data)
  FM_CACHE      window caches built by build_*.py and cache_fourier.py (default: experiments/fm_mini/cache, git-ignored)
  FM_RUNS       run folders with checkpoints and evaluation tables (default: experiments/fm_mini/runs, git-ignored)
"""
import os
import sys
from pathlib import Path

FM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FM_DIR.parent/'decomposition'/'benchmark'))
from data import ROOT, VER0_DIR  # noqa: E402,F401

CACHE = Path(os.environ.get('FM_CACHE', FM_DIR/'cache'))
RUNS = Path(os.environ.get('FM_RUNS', FM_DIR/'runs'))
