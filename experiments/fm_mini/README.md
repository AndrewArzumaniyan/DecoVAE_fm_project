# Mini foundation-model experiments: computed seasonality and latent-mixture training

Code, results and figures of our preliminary report (course *Models of Sequential Data*).
Small probabilistic forecasting models (0.1–1M parameters) are pretrained on 9 sources of the corpus and evaluated
on new series of the same sources (**ID**) and zero-shot on 6 held-out sources (**OOD**). We compare the original
DecoVAE-FM (`decovaefm/` in the repository root, used unchanged), plain patch Transformers of the same size, and our
period-aware decomposition: the seasonal part is computed from each window and the network learns the rest.

**Start here:** [`notebooks/results_overview.ipynb`](../../notebooks/results_overview.ipynb) shows all tables and
figures of the report (saved with outputs, so it renders on GitHub). The same tables as text are in
[`results/tables.md`](results/tables.md).

## Main results

| | OOD skill | ID skill |
|---|---|---|
| TF-6L-MixC (plain Transformer, size of DV-FM) | 0.747 ± 0.015 | 0.418 ± 0.006 |
| **PD-MixC (ours)** | **0.649 ± 0.004** | 0.430 ± 0.006 |
| ratio PD-MixC / TF-6L-MixC, 95% CI | 0.868 [0.850; 0.885] | 1.029 [1.007; 1.058] |

Skill = geometric mean over sources of mean CRPS / mean CRPS of seasonal-naive (lower is better); mean ± sd over 5
seeds; CI by a paired cluster bootstrap over series. Also in the tables: the same comparison for the NLL and -Mix
losses, the original DV-FM, KernelSynth synthetic data, a larger scale, calibration, the synthetic lab and ablations.

## Names

| report | arm (`train.py --arm`) | what it is |
|---|---|---|
| DV-FM | `A` | original DecoVAE-FM: trend Transformer, seasonal and noise streams |
| TF-2L / TF-6L | `N --n-layers 2` / `N6` | the trend patch Transformer alone on the raw window, 2 / 6 layers |
| PD | `Dcomb` | period-aware decomposition: computed Fourier seasonal part, deseasonalised trend input with level anchor, variance features |
| PD-Mix / PD-MixC | `Dmix` / `Dmc1` | PD trained with the log score of the 8-draw latent mixture / plus almost-fair CRPS |
| TF-6L-Mix / TF-6L-MixC | `N6mix` / `N6mc1` | TF-6L with the same losses |
| PD-CVAE-S / PD-CVAE-G | `Dzsh` / `Dzgh` | PD + structured (amplitude / phase / level) or general latent trained by ELBO |
| ablation arms | `Cf`, `Df`, `Cf0`, `Dd`, `Dda`, `Dfs`, `Cs1`, `Cs2`, `Dmc1s`, `Dhp*`, `DcombCAstd`, `DmixCAstd` | see the docstrings of `trend_input_arms.py`, `seasonal_arms.py`, `score_arms.py`, `hp_arms.py` |

Run ids in `results/` are `<series>_<arm>[_variant]_s<seed>` (e.g. `srv10_Dmc1_s3`); `paper_tables.py` maps every
table row to its run families. `results/runs.csv` holds the settings of every run.

## Layout

| file | role |
|---|---|
| `fm_paths.py` | locations of data, caches and runs; puts the shared modules of `../decomposition/benchmark` on `sys.path` |
| `build_data.py` | main corpus: windows of 9 training and 6 held-out sources |
| `cache_fourier.py`, `seasonal_arms.py` | per-window period estimation + Fourier fit, and window features, cached once per corpus |
| `build_scaled.py`, `build_nested.py`, `derive_prefix.py` | larger corpora (4×, 16×, 64× windows of the same sources) |
| `build_kernelsynth.py` | KernelSynth generator (as in Chronos) and mixed caches |
| `build_lab.py`, `lab_eval.py` | synthetic decomposition lab with known components |
| `build_synth_cvae.py`, `cvae_diag.py` | synthetic data with known shifts of amplitude / phase / level; latent diagnostics |
| `build_asinh.py` | asinh-space copy of a cache |
| `build_loso.py` | leave-one-source-out folds (experiment not finished, see below) |
| `train.py` | training of all arms (reuses `../decomposition/benchmark/runner.py`); also scores test_ood |
| `trend_input_arms.py`, `seasonal_arms.py`, `score_arms.py`, `cvae_arms.py`, `hp_arms.py` | model variants |
| `eval_space.py` | evaluation used in all tables: CRPS from 128 samples, energy / variogram scores, PIT |
| `baselines_eval.py`, `es_fourier_baseline.py` | statistical baselines (seasonal-naive is the skill reference) |
| `collect_results.py` | run folders → the small files in `results/` |
| `paper_tables.py` | `results/` → all tables (`results/tables.md`) |
| `make_examples.py`, `figure_data.py` | example forecasts and the data of the forecast figure |
| `results/` | per-window scores, calibration, training curves, run settings, baselines, lab rows, figure data (≈20 MB) |

## Setup

From the repository root:

```bash
uv sync --locked --group experiments
source .venv/bin/activate
```

Python 3.11, torch 2.5.1, numpy 2.4.6, scipy 1.17.1, statsmodels 0.15 as pinned in `uv.lock`. Training needs a CUDA
GPU (we used an RTX 4090; one run on the base corpus takes 10–40 minutes, a scale run about 1.5 hours). Tables, the notebook and
the CPU checks need no GPU and no data:

```bash
cd experiments/fm_mini
python paper_tables.py                                   # prints all tables, writes results/tables.md
uv run --with jupyterlab jupyter lab ../../notebooks/results_overview.ipynb
```

## Data

The pretraining corpus is produced by `convert_data.sh` (`decovaefm/data/unified_converter.py`) from public Monash,
M4, Chronos and LOTSA files; see [`../decomposition/README.md`](../decomposition/README.md#data) for its sources and
split. Point `DECOVAE_VER0` to the folder with `manifest.json`. Caches and runs go to `experiments/fm_mini/cache` and
`experiments/fm_mini/runs` (git-ignored; override with `FM_CACHE`, `FM_RUNS`).

Training sources: London_Smart_Meters, Q-TRAFFIC, azure_vm_traces_2017, borg_cluster_data_2011, largest_2017,
weatherbench_weekly, ushcn_daily, Wind_Farms, mexico_city_bikes. Held-out (OOD) sources: Electricity, Solar,
KDD_Cup_2018, Pedestrian_Counts, M4_Daily, Uber_TLC. 17,196 training windows, 965 ID and 1,536 OOD test windows,
context 512, horizon 96.

## Reproducing

**Main corpus.** The published test_ood split was drawn by a second invocation, after the other splits existed, so
run the builder twice:

```bash
cd experiments/fm_mini
python build_data.py                     # writes cache/fm_mini_<hash>/{train,val,test_id,test_ood}.npz
rm cache/fm_mini_*/test_ood.npz && python build_data.py
C=$(ls -d cache/fm_mini_*)
OMP_NUM_THREADS=1 python cache_fourier.py $C --workers 8 --clip 20
OMP_NUM_THREADS=1 python seasonal_arms.py $C --workers 8
python baselines_eval.py $C && python es_fourier_baseline.py $C
```

We checked this in a clean checkout: the four window files are byte-identical to the ones used for the report
(sha256 of test_ood.npz `aa9ed15c…`), and the Fourier cache agrees to 1e-13.

**Training and evaluation.** All runs of the main tables use the same protocol:

```bash
python train.py --cache $C --arm Dmc1 --seed 0 --run-id Dmc1_s0 \
    --batch 256 --lr 1e-3 --recon-weight 0.2 --clip 20 --epochs 472 --minutes 180 --val-every 8 --fast-val
python eval_space.py runs/Dmc1_s0          # space_eval_{test_id,test_ood}.csv + PIT, the input of the tables
```

Replace `--arm` by `N6mc1`, `Dmix`, `N6mix`, `Dcomb`, `N6`, `A`, … and use seeds 0–4 (TF-2L: `--arm N --n-layers 2`). Exceptions: the scale runs use
the 16× corpus (`build_nested.py --base $C`, then `cache_fourier.py`, `derive_prefix.py`), `--d-model 128`,
`--val-every 1` and, for N6mc1, `--n-layers 5`; the KernelSynth runs use the mixed caches of `build_kernelsynth.py`;
the lab runs use `build_lab.py`; exact settings of every run are in `results/runs.csv`. Evaluating a saved checkpoint
is deterministic: re-evaluating the report's PD-MixC seed-0 checkpoint with this code gives identical window scores.
Retraining on a GPU is not bit-exact (cuBLAS kernel choice), so retrained models agree up to seed-level noise.

**Tables.** `python collect_results.py --runs runs --out results --baselines runs/baseline_rows.csv
runs/baseline_rows_es.csv`, then `python paper_tables.py`.

## Incomplete experiments and limitations

- **Leave-one-source-out** over all 15 sources (`build_loso.py`): running at the time of the report; folds are not
  included in `results/`.
- Not yet in this folder: post-hoc recalibration of the intervals (mentioned in the report as a baseline) and the
  script of the environment check; the latter's training runs are included (`envchk_*` in `results/`).
- `results/fig_lab_windows.npz` (lab decomposition figure) was dumped from the `lab_A_s0` / `lab_Dcomb_s0` checkpoints
  by a one-off script that is not included.
- Runs up to the report were trained in a separate environment with numpy 1.26 / scipy 1.13 / torch 2.5.1+cu121
  (versions in `results/runs.csv`). Checkpoint evaluation is bit-identical in both environments; retraining in the
  pinned environment gives a ratio of 0.864 vs 0.868, within the seed spread.
- No external baselines trained with our protocol (DecoVAE, K²VAE, a pretrained FM that never saw the held-out
  sources); the scale is small (≤ 1M parameters, 9 sources).
