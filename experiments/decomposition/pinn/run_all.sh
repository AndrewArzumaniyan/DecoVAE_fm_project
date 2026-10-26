#!/bin/bash
# Full grid: 8 structural variants x 3 seeds, then HP-weight sweep on V3 and V0+HP.
export OMP_NUM_THREADS=1
cd "$(dirname "$0")"
for s in 0 1 2; do
  for v in V0_original V1_period_in V2_hard_periodic V2det_fixed_seas V3_hard+HP V4_soft_periodic V5_hard+teacher V0_original+HP; do
    python train.py "$v" "$s" 2>&1 | grep -v Warn | cut -c1-200
  done
  for g in 0.1 10 100; do
    python train.py V3_hard+HP "$s" "$g" 2>&1 | grep -v Warn | cut -c1-200
  done
  python train.py V0_original+HP "$s" 10 2>&1 | grep -v Warn | cut -c1-200
done
echo ALL_DONE
