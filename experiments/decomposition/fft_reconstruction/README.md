# FFT reconstruction only

Compare the current repository model with two changes to its final seasonal reconstruction:

| Variant | Frequency | Coefficient convention |
|---|---|---|
| `original` | `round(k * (T+H) / T) / (T+H)` | Native `irfft(T+H)` |
| `amplitude` | Same rounded bins | Multiply coefficients by `(T+H)/T` before native `irfft` |
| `exact_frequency` | Original `k/T`, directly evaluated | Length-T rFFT: divide by T; multiplicity 1 for DC/Nyquist, 2 otherwise |

`model.py` subclasses the current `DecoVAEFM` and overrides **only**
`_reconstruct_time_from_freq_tokens`. Its forward, encoders, learned MA decomposition, all learned
decoders and variance head remain native. There are no new parameters, true-period inputs,
deterministic seasonal coefficients, structural losses or teacher targets.

Exact-frequency reconstruction is the **T-periodic extension of the same top-K FFT bins**.
It does not estimate off-grid periods or remove spectral leakage. It preserves a grid-aligned
sinusoid when its period divides T; the existing rounded remap can distort that very sinusoid.
For example, period24 does not divide512: its true value at t+512 differs from its value at t,
whereas the `exact_frequency` seasonal necessarily repeats. This hard constraint can hurt forecasts
on the base periods12/24/48/96 even after eliminating the rounded transfer.
At a fixed horizon, the amplitude-only change is a constant factor that a learned seasonal head
can absorb. Correcting the identity-decoder error therefore need not improve a trained forecast.

## Training protocol

- T=512; H=96/128/256; seeds 0/1/2; 27 runs of 30 epochs. The default runs three independent CPU
  processes, each with one PyTorch thread and its own RNG. `--jobs 1` executes serially.
- Data: the existing `benchmark/data.py` generator, 45 base scenarios, 2048 train / 256 validation /
  512 base test. Test additionally contains 12 held-out diagnostic cases with 32 series each:
  grid periods 32/64/128/256, off-grid 19.7/37.1/168, dual 24+168 and 17.3+96, amplitude growth,
  noise-only and trend+noise controls. Diagnostics are absent from training and validation.
- Generate full paths of length 768 once, then slice the same paths for every H. Each window uses
  **linear normalization from its own 512 context points**, exactly as in the existing benchmark.
  Generation seeds are train719 / validation100719 / test200719; diagnostic streams are separate.
- Native d_model64, patch=stride16, enc_layers=dec_layers2, heads4, k_top16, kernels[5,9,13,25].
  Initial state hashes, mini-batch order and dropout/latent RNG streams match within seed/H pairs.
  Learned decomposition weights can diverge during training; the input algorithm is identical.
- Loss is imported directly from **`benchmark.runner.loss`**, unchanged: recon NLL + forecast NLL
  + 0.01 KL, including that function's complex-KL convention. This is the benchmark loss, not a
  newly copied or altered Lightning loss. AdamW0.001, batch32, clip1.
- Fixed 30-epoch budget for all variants; select the minimum validation CRPS checkpoint (32
  latent+observation draws each epoch). Test uses 128 draws, with the same seed1729 in each arm.
  Validation RNG is isolated from training. No test-based checkpoint selection or early stopping.

This is a controlled synthetic ablation, not the larger 8192-series, max60-epoch `night` benchmark.
Each horizon is trained separately; zero-shot transfer of one trained model to other horizons is not tested.
The local published runtime is recorded in provenance (Python3.13 / torch2.8 CPU), and differs
from the repository's Python3.11 / torch2.5.1 lock. Within this grid the runtime is the same.

## Reproduction

Run from the repository root using the environment from the parent README:

```bash
python -m unittest discover -s experiments/decomposition/fft_reconstruction -p 'test_*.py' -v
python experiments/decomposition/fft_reconstruction/identity_check.py
python experiments/decomposition/fft_reconstruction/train.py --profile smoke --horizons 96 256 --seeds 0
python experiments/decomposition/fft_reconstruction/train.py --profile full
python experiments/decomposition/fft_reconstruction/report.py
```

The report requires `nbformat` and IPython to save an executed notebook. It runs in-process,
without opening a Jupyter socket. In a minimal environment, install `nbformat` before reporting.

`DECOVAE_RUNS` relocates raw run outputs. Defaults:

- `../runs/fft_reconstruction_full/`: provenance, snapshots, data hashes and NPZ splits; each arm's
  best checkpoint, history, status, per-series metrics and mean component/forecast arrays.
- `../results/fft_reconstruction_*.csv`: identity errors, per-seed/aggregated scores, paired
  differences with 95% row-bootstrap intervals (2000 resamples within each training seed), run metadata,
  and the true seasonal energy retained in the encoder's selected FFT bins (post-training diagnostic only).
- `../fft_reconstruction_summary.ipynb`: tables and figures, saved with executed outputs.
- `FINDINGS.md`: numerical report of the full grid.

Negative paired deltas mean improvement. Bootstrap intervals represent test-series sampling
uncertainty within a seed, not uncertainty over future training seeds. Diagnostic comparisons
are exploratory and have no multiple-comparison correction. Forecast/component errors use the
whole context's std; the identity check uses the true seasonal std of the evaluated segment.

Re-running the same run id reuses completed arms only when config, recorded source hashes and
runtime match. An incomplete arm restarts from initialization. Use a new `--run-id` after changes.
No source in `decovaefm/` or the existing benchmark is modified by this experiment.
