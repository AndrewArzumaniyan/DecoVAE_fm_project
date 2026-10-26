# Findings

Mechanism-level results for the `decovaefm` architecture. Unless stated otherwise: synthetic data
with known components, context T = 512, horizon L = 96, three training seeds reported separately,
paired row-bootstrap intervals (2,000 resamples) within a seed. Tables are in [results/](results/), figures in [summary.ipynb](summary.ipynb);
how to reproduce them is in [README.md](README.md). Arms A/B/C/D/Cf/Df are defined in the README.

These results do not include the decisive comparison of the project plan (DecoVAE-FM against
K²VAE and zero-shot TSFMs on real benchmarks). They concern the internal mechanisms that this
comparison depends on.

## 1. The original input decomposition does not separate trend and seasonality

With the learnable multi-scale moving average, the trend branch carries almost the whole signal.

- In a trained model on synthetic data with a strong seasonal component, the seasonal branch
  carries a median share of 0.002 of the signal (ground truth 0.95) (`pinn_variants.csv`,
  `V0_original`).
- In the A/B arms, the input components have NRMSE 0.64 (trend) and 0.67 (seasonal) against the
  truth (`v1_components_by_seed.csv`).

Per-window period estimation with a Fourier seasonal at non-integer periods and an HP trend
(`period_decomp/`) recovers the components without any training:

| median RMSE / std(x), single season | trend | seasonal (context) | seasonal (horizon) |
|---|---|---|---|
| original, uniform weights | 0.476 | 0.493 | 0.555 |
| original, trained weights | 0.536 | 0.514 | 0.579 |
| period-aware, estimated period | 0.183 | 0.109 | 0.124 |
| period-aware, true period | 0.162 | 0.092 | 0.092 |

(`period_decomp_synthetic.csv`; with two seasons, seasonal horizon error is 0.203 vs 0.710.)

Feeding this decomposition to the model (C/D) gives input NRMSE 0.084 (trend) and 0.034
(seasonal), and the seasonal branch is used (share 0.87, `V1_period_in`).

## 2. The original seasonal decoder limits the forecast for most periods

The decoder scatters the top-K rFFT bins of the length-T context onto the rFFT grid of length T+L
with the index k·(T+L)/T rounded to an integer, and inverts with irfft of length T+L. An identity
check (true coefficients, no training; `checks/seasonal_decoder_check.py`,
`seasonal_decoder_identity_check.csv`) separates two effects:

- **Amplitude.** The coefficients are not rescaled from length T to T+L. For p = 32, where every
  remapped index is an integer, this is the only error: NRMSE 0.16 = 1 − T/(T+L); with rescaling 0.00.
- **Frequency rounding.** The transfer is exact only if k·(T+L)/T is an integer for every kept bin.
  Otherwise the frequency of the extrapolated seasonal shifts. For p = 64 the fundamental bin 8
  maps to 9.5; the error is 1.07 on the context and 1.66 on the horizon, while the same bins
  evaluated at their exact frequency give 0.00.

| period (K = 32) | 12, 24, 48, 96 | 19.7 | 30.4 | 17.3, 37.1, 64, 144, 168 |
|---|---|---|---|---|
| NRMSE context / horizon | 0.19–0.21 / 0.25–0.29 | 0.51 / 0.90 | 0.62 / 1.04 | 0.84–1.07 / 1.66–1.90 |

(A zero forecast has NRMSE 1.0.) In trained C/D models the decoder turns accurate input
components into poor seasonal estimates: on off-grid and dual periods the seasonal NRMSE grows
from 0.03–0.13 at the input to 0.26–0.96 after decoding (`diagnostic_components_by_case.csv`).
On the seasonal block of the `synthetic_v2` test set this makes C and D worse than A
(CRPS 0.444 and 0.447 vs 0.212, `v2_test_by_block.csv`).

## 3. A continuous-frequency Fourier decoder removes that failure, but does not beat A overall

Cf/Df replace the seasonal decoder with a Fourier series at the estimated periods plus a learned
correction. Criteria were fixed before training (`fourier_criteria.csv`):

| criterion | result |
|---|---|
| K1: decoded seasonal NRMSE ≤ 2 × input (3 key cases) | met: ratio 1.00 |
| K2: Cf − A CRPS interval below 0 in ≥ 2 of 3 seeds | met on the 3 key cases (3/3 seeds); not met on the v2 seasonal block (1/3) |
| K2 no-harm: no case where Cf is worse than A in all 3 seeds | not met: 5 cases |
| K3: v2 seasonal-block CRPS ≤ 0.21 | not met: 0.2103 |

- Where the original decoder failed, Cf is better than A in every seed: period 37.1 (−0.013 to
  −0.037 CRPS) and dual seasonality 17.3 + 96 (−0.05 to −0.10). On the seasonal block Cf and A are
  on par (0.210 vs 0.212; B 0.208).
- Cf is worse than A in all seeds when the amplitude changes over the window (amplitude growth,
  +0.006 to +0.013) and for dual seasonality 24 + 168 (+0.044 to +0.050). The deterministic
  Fourier extrapolation alone is also poor there (`diagnostic_mean_mse.csv`, `period_det`), so
  these losses come from the period estimate (about three cycles of p = 168 in T = 512) and the
  stationary Fourier form, not from the decoder.
- On the iid block, Cf has lower CRPS than A in every seed (0.205 vs 0.233), with slightly lower
  coverage (0.879 vs 0.895).
- The learned correction is not used: with the correction set to zero (Cf0) every case score is
  unchanged to the fourth decimal.

## 4. The noise branch has no consistent effect on Gaussian synthetic data

The noise arms keep the σ-head and the NLL; only the noise latent Z_n is removed (B, D, Df).

- First synthetic profile (`runner.py --profile night`: the 45 base scenarios), G = CRPS(without) −
  CRPS(with), per seed (CRPS ≈ 0.22):
  original decomposition −0.0019, −0.0016, +0.0001; period-aware +0.0030, −0.0010, +0.0147
  (`v1_noise_branch_effects.csv`). The sign changes with the seed.
- Cf vs Df: |difference| ≤ 0.012 per case. The only effect with the same sign in all seeds is on
  the iid block, Cf − Df = −0.003 to −0.007 (`v2_test_paired_by_seed.csv`).
- On the first profile every arm reaches 0.86–0.91 coverage at the 90 % level, so these data
  cannot discriminate a better noise model.
- In the debugging example of `notebooks/debug_training.ipynb`, the predicted σ is driven by the
  trend latent: shuffling Z_n across series changes log σ by 0.006 on average, shuffling Z_t by
  0.55–0.60 (`checks/notebook_repro.py`).

## 5. How the predicted variance responds to input changes

Paired interventions on 64 series (`benchmark/interventions.py`, `interventions_by_arm.csv`;
median over seeds of the per-model median ratio; D has two seeds):

| intervention | ideal | A | B | C | D | Cf | Df |
|---|---|---|---|---|---|---|---|
| noise × 1.5 | 2.25 | 1.99 | 1.92 | 1.71 | 1.72 | 2.11 | 2.07 |
| seasonal × 1.5 | 1.00 | 1.16 | 1.16 | 1.27 | 1.39 | 1.04 | 1.13 |
| trend slope + 0.003 | 1.00 | 1.14 | 1.16 | 1.08 | 1.09 | 1.09 | 1.04 |

All arms react to the noise level, with or without Z_n. Seasonal amplitude leaks into the
predicted variance, least in Cf. Each window must be normalised by its own context statistics in
such tests: normalising the modified window with the base statistics produced spurious ratios of
100× and more in an earlier version of the protocol.

## 6. The room for a better noise model is small on realistic volatility

For noise with a known volatility state (`benchmark/h1_data.py`), a Gaussian forecast with the true
conditional variance is compared with a constant σ estimated from the context; both use the true
mean (`h1_headroom_*.csv`, horizon steps 1–32):

| CRPS gain of the true variance | GARCH | stochastic volatility | regime switching |
|---|---|---|---|
| realistic parameters | 1.1 % | 2.6 % | 2.1 % |
| strong persistence and volatility | 3.8 % | 14.2 % | 6.1 % |

A σ from the last 32 noise values already captures most of this (stochastic volatility, strong:
4.6 % left). A test of the noise branch therefore needs data like the strong profile and a control
that gives the σ-head any unstructured high-frequency path, not only "remove Z_n".

## 7. Zero-shot transfer to real windows is weak and source-dependent

Models trained only on `synthetic_v2`, evaluated on fixed test windows of the pretraining corpus
(5–8 windows per source; `real_transfer_by_seed.csv`):

- M4_Daily: 90 % coverage 0.17–0.23 for A/B, 0.33–0.62 for C/Cf, 0.63–0.80 for D/Df.
- Electricity: normalised CRPS 0.33–0.34 for A, 0.27–0.29 for Cf, 0.37–0.40 for C/D.
- London smart meters and Q-TRAFFIC: no arm stands out.

With this few windows these numbers describe tendencies, not established differences.

## 8. Isolating FFT reconstruction alone does not restore decomposition

The [FFT-only ablation](fft_reconstruction/README.md) trains the native model with three final
reconstruction operations: original, amplitude rescaling only, and amplitude correction plus
evaluation at the original frequencies k/T without rounding. The original learned MA, encoders,
all heads, parameter count and `benchmark.runner.loss` are unchanged. Initial weights and batch
orders match within each seed/horizon pair; there is no period estimator or fixed seasonal baseline.

27 runs: horizons 96/128/256 × seeds 0/1/2 × three variants, 30 epochs each; 2048 training and 256
validation series from the 45 base scenarios. Test includes 512 base series and 12 held-out
diagnostic cases × 32 series. Each horizon is trained separately, using prefixes of shared paths.
This local series uses Python 3.13.5 / torch 2.8.0 on CPU, unlike the earlier locked benchmark.

| horizon | original normalized CRPS | amplitude only | exact k/T |
|---|---|---|---|
| 96 | 0.31994 | 0.32186 | 0.32784 |
| 128 | 0.33286 | 0.33490 | 0.34044 |
| 256 | 0.36098 | 0.36173 | 0.38142 |

The identity check confirms amplitude and rounded-frequency errors. Nevertheless, neither change
improves mean base-test CRPS at any horizon. Exact k/T is worse in all three seeds at every horizon;
amplitude-only improves two seeds at H256 but worsens the third enough to offset the gain.
No horizon has a paired CRPS interval entirely below zero in at least two of three seeds.

Decoded seasonal projection gains remain 0.022–0.144 (an exact match would be 1). For periods
64/128/256, the true seasonal energy retained in the encoder's top-K bins averages 0.23% / near
zero / near zero, while the decoded trend carries the seasonal signal (projection gains near 1).
The decoder cannot recover frequencies absent from the seasonal representation.

At a fixed horizon the amplitude factor can be absorbed by the learned final complex-linear head;
this is verified on a trained checkpoint to 2.38e-7 maximum output error. Exact k/T also forces a
T-periodic seasonal: s(t+512)=s(t), whereas the base periods 12/24/48/96 do not divide512. Eliminating
rounded transfer therefore does not eliminate the off-grid extrapolation limitation.

Per-seed scores, paired intervals and diagnostics: `results/fft_reconstruction_*.csv`.
[Executed notebook](fft_reconstruction_summary.ipynb); [detailed report](fft_reconstruction/FINDINGS.md).
These results concern this synthetic training profile, not real-data accuracy or unseen-horizon transfer.

## Limitations

- Synthetic data mostly follow the additive trend + seasonal + Gaussian noise form the model
  assumes; they test mechanisms, not real-world accuracy.
- Three training seeds per arm; one cell (D, seed 2, `synthetic_v2`) is missing and C seed 2 was
  stopped at epoch 59.
- The pretraining corpus is split by series, not by time; the real-data check uses few windows.
- Calibration is reported through coverage and CRPS only; PIT-based calibration errors are not
  computed yet.
- Training on a GPU is not bit-exact; retrained models reproduce the metrics up to small
  differences.
