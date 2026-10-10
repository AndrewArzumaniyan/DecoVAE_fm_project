#!/bin/bash
export OMP_NUM_THREADS=1
cd "$(dirname "$0")"
for s in 0 1 2; do
  for v in V5b_period_in+teacher V5c_original+teacher V2r_det+learned_resid; do
    python train.py "$v" "$s" 2>&1 | grep -v Warn | cut -c1-120
  done
done
echo ALL_DONE
