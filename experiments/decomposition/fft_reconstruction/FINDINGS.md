# FFT-only reconstruction ablation

Completed: 27 runs, 3 variants × 3 horizons (96/128/256) × 3 training seeds (0/1/2), 30 epochs each.
Native DecoVAEFM; only `_reconstruct_time_from_freq_tokens` differs. Original MA, learned encoders,
trend/seasonal/variance heads, noise latent and unchanged `benchmark.runner.loss` are shared.
All pairs have identical initial parameter hashes and batch orders. The learned MA weights remain trainable.

## Base test: mean ± SD over training seeds

CRPS and NRMSE are in context-standardized units; lower is better. SD is not a confidence interval.

| H | Variant | CRPS | Forecast NRMSE | Coverage90 | Seasonal horizon NRMSE | Seasonal gain (context) |
|---|---|---|---|---|---|---|
| 96 | original | 0.31994 ± 0.00270 | 0.55020 | 0.9032 | 0.62820 | 0.14351 |
| 96 | amplitude | 0.32186 ± 0.00080 | 0.55318 | 0.8979 | 0.62892 | 0.13786 |
| 96 | exact_frequency | 0.32784 ± 0.00156 | 0.56288 | 0.9056 | 0.69188 | 0.09823 |
| 128 | original | 0.33286 ± 0.00750 | 0.56840 | 0.8991 | 0.66888 | 0.02564 |
| 128 | amplitude | 0.33490 ± 0.00738 | 0.57103 | 0.9020 | 0.66521 | 0.02239 |
| 128 | exact_frequency | 0.34044 ± 0.00866 | 0.58237 | 0.8938 | 0.69129 | 0.09393 |
| 256 | original | 0.36098 ± 0.00338 | 0.61053 | 0.9030 | 0.65536 | 0.05365 |
| 256 | amplitude | 0.36173 ± 0.00962 | 0.61076 | 0.9005 | 0.65720 | 0.05729 |
| 256 | exact_frequency | 0.38142 ± 0.01671 | 0.64251 | 0.8922 | 0.70524 | 0.09881 |

## Paired forecast CRPS differences

Delta = corrected − original. Negative is better. Intervals are paired row-bootstrap (2000 resamples)
inside a training seed; they do not quantify uncertainty over future training seeds.

| H | Variant | Mean delta over seeds | Improved seeds | Seeds with CI entirely below zero |
|---|---|---|---|---|
| 96 | amplitude | +0.001912 | 0/3 | 0/3 |
| 96 | exact_frequency | +0.007901 | 0/3 | 0/3 |
| 128 | amplitude | +0.002044 | 0/3 | 0/3 |
| 128 | exact_frequency | +0.007580 | 0/3 | 0/3 |
| 256 | amplitude | +0.000749 | 2/3 | 1/3 |
| 256 | exact_frequency | +0.020446 | 0/3 | 0/3 |

## Observed results

- `amplitude`: base-test improvement with a bootstrap interval entirely below zero in at least 2/3 seeds at horizons: none.
  Diagnostic case × horizon pairs worse in all three seeds: 3 / 36 (exploratory signs, not multiplicity-corrected significance).
- `exact_frequency`: base-test improvement with a bootstrap interval entirely below zero in at least 2/3 seeds at horizons: none.
  Diagnostic case × horizon pairs worse in all three seeds: 6 / 36 (exploratory signs, not multiplicity-corrected significance).
- Average decoded seasonal projection gains on the base test range from 0.0224 to 0.1435; an exact component match would have gain 1.
  None of these decoder-only variants restores full seasonal component recovery on this training profile.
- `period_64`: mean fraction of true seasonal FFT variance retained in the encoder's top-K bins is 0.0023; decoded trend's seasonal projection gain is 0.9995 (ideal zero), averaged over this grid.
- `period_128`: mean fraction of true seasonal FFT variance retained in the encoder's top-K bins is 0.0000; decoded trend's seasonal projection gain is 1.0051 (ideal zero), averaged over this grid.
- `period_256`: mean fraction of true seasonal FFT variance retained in the encoder's top-K bins is 0.0000; decoded trend's seasonal projection gain is 1.0107 (ideal zero), averaged over this grid.
- In the trained original H96/seed0 checkpoint, scaling the final complex-linear seasonal head by 512/608 compensates amplitude-only reconstruction: maximum output difference 2.38e-07 on eight fixed test contexts with paired latent draws.

## Interpretation and scope

- The identity check confirms the amplitude convention and bin-rounding defects without training.
  Exact-frequency evaluation reconstructs periods dividing T=512 up to numerical precision.
- Exact-frequency evaluation is a T-periodic extension of the same top-K bins, not a true-period estimator.
  Off-grid seasonality can still extrapolate poorly. A decoder-only fix does not remove this limitation.
  In particular, the base positive periods 12/24/48/96 do not divide 512. The exact-k/T decoder forces
  seasonal(t+512)=seasonal(t), which the true seasonal does not satisfy; the trend head can compensate
  in the total mean, but this does not restore true component recovery.
- For a fixed horizon, amplitude-only correction is a constant rescaling which the learned seasonal head
  can absorb. The identity error is real, but a forecast improvement after training is not guaranteed.
- Trained performance and component recovery must be assessed separately using the table above and diagnostics.
  An operator defect does not by itself establish that correcting it fixes learned decomposition.
- Per-case diagnostics, paired intervals and three horizon segments are in results/fft_reconstruction_*.csv.
  Diagnostic comparisons are exploratory and have no multiple-comparison correction.

## Reproduction and limitations

2048 train / 256 validation / 512 base test + 12 × 32 held-out diagnostic series. Linear context-only
normalization, AdamW 0.001, batch32, clip1. Best validation-CRPS checkpoint from a fixed 30-epoch budget;
32 validation and 128 test latent + observation draws. No test metric is used to select checkpoints.
Same underlying paths are sliced for all horizons. Series seeds: 719 / 100719 / 200719.
Synthetic data only; three training seeds; this is not the larger existing `night` benchmark.
Each horizon is trained separately. Generalization of one trained model to an unseen horizon is not tested.
Local runtime: Python 3.13.5, torch 2.8.0, CPU; differs from uv.lock.
Provenance, data hashes, source snapshots, checkpoints, histories and per-series outputs are under
`runs/fft_reconstruction_full/` (Git-ignored). Summaries and the executed notebook are tracked artifacts.
