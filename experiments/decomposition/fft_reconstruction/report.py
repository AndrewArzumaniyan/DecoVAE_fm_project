"""Summarize completed FFT-only runs; save CSVs and an executed figure notebook."""
import argparse
import json
from pathlib import Path
import sys

import nbformat
import numpy as np
import pandas as pd
import torch

from datasets import EXPERIMENTS
from model import FFTReconstructionArm
sys.path.insert(0, str(EXPERIMENTS))
from paths import RUNS_DIR
from identity_check import identity_table


@torch.no_grad()
def topk_signal_retention(root, runs):
    """Post-training diagnostic only: true seasonal FFT energy in selected bins.

    True components are used solely to score the selected representation, never as inputs.
    The encoder is evaluated on its ordinary MA residual. Center truth to measure variance.
    """
    torch.set_num_threads(1)
    raw = dict(np.load(root / "test.npz"))
    rows = []
    for run in runs:
        h, seed, variant = run["horizon"], run["seed"], run["variant"]
        checkpoint = torch.load(root / f"h{h}_s{seed}_{variant}" / "best.pt", weights_only=False)
        prediction = np.load(root / f"h{h}_s{seed}_{variant}" / "predictions.npz")["trend"][:, :512]
        true_trend = (raw["trend"][:, :512] - raw["mean"]) / raw["scale"]
        true_seasonal = raw["seasonal"][:, :512] / raw["scale"]
        centered = true_seasonal - true_seasonal.mean(1, keepdims=True)
        denominator = (centered ** 2).sum(1)
        trend_gain = ((prediction - true_trend) * centered).sum(1) / np.maximum(denominator, 1e-12)
        trend_gain[denominator < 1e-10] = np.nan
        model = FFTReconstructionArm(variant, pred_len=h, **checkpoint["model_kwargs"]).eval()
        model.load_state_dict(checkpoint["model"])
        for start in range(0, len(raw["z"]), 32):
            sl = slice(start, start + 32)
            x = torch.from_numpy(raw["z"][sl, :512])
            indices = model.encode(x, n_target=h // 16)["topk_idx"]
            truth = torch.from_numpy(raw["seasonal"][sl, :512])
            truth = truth - truth.mean(1, keepdim=True)
            power = torch.fft.rfft(truth).abs().square()
            weights = torch.full((power.shape[-1],), 2.)
            weights[0] = weights[-1] = 1.
            power = power * weights
            total = power.sum(1)
            retained = power.gather(1, indices).sum(1) / total.clamp_min(1e-12)
            retained[total < 1e-10] = float("nan")
            for j, value in enumerate(retained):
                rows.append(dict(horizon=h, seed=seed, variant=variant,
                                 case=str(raw["case"][start + j]), topk_true_seasonal_energy=float(value),
                                 decoded_trend_seasonal_gain=float(trend_gain[start + j])))
    return pd.DataFrame(rows).groupby(["horizon", "seed", "variant", "case"], as_index=False).mean()


def paired_summary(rows):
    rng = np.random.default_rng(991)
    output = []
    keys = ["horizon", "seed", "case"]
    for (horizon, seed, case), data in rows.groupby(keys):
        baseline = data[data.variant == "original"].set_index("series_id")
        for variant in ("amplitude", "exact_frequency"):
            after = data[data.variant == variant].set_index("series_id")
            if len(after) != len(baseline) or set(after.index) != set(baseline.index):
                raise ValueError("Paired comparison requires identical test series")
            after = after.loc[baseline.index]
            for metric in ("ncrps", "nrmse", "seasonal_hor_nrmse"):
                values = (after[metric] - baseline[metric]).to_numpy()
                boot = rng.choice(values, size=(2000, len(values)), replace=True).mean(1)
                low, high = np.quantile(boot, [.025, .975])
                output.append(dict(horizon=horizon, seed=seed, case=case, variant=variant,
                                   metric=metric, n=len(values), delta=float(values.mean()),
                                   ci_low=float(low), ci_high=float(high),
                                   improvement_pct=float(-100 * values.mean() / baseline[metric].mean())))
    return pd.DataFrame(output)


def build_notebook(root, summary, paired):
    cells = []
    md = lambda source: cells.append(nbformat.v4.new_markdown_cell(source))
    code = lambda source: cells.append(nbformat.v4.new_code_cell(source))
    md("""# FFT reconstruction: isolated ablation

Three variants of the **current DecoVAEFM**, differing only in the final conversion
of predicted complex coefficients into a seasonal curve:

1. `original`: the current FFT bin remapping and `irfft(T+H)`.
2. `amplitude`: the same remapping, with coefficients multiplied by `(T+H)/T`.
3. `exact_frequency`: the same coefficients evaluated directly at frequencies `k/T`;
   scale `1/T`, weight 1 for DC/Nyquist and 2 for all other frequencies.

The encoders, input decomposition, heads, parameter count, and loss are identical.
No precomputed seasonal components or period estimates were added. The main package code was unchanged.

Exact `k/T` preserves the original FFT frequencies but **does not estimate true non-integer periods**:
the extension remains T-periodic. All components are measured in linearly normalized
coordinates. NRMSE is expressed in units of the standard deviation of the *full series context*.
In the identity check, the unit is the standard deviation of the true seasonal component in the corresponding segment.
""")
    code("""from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

cwd = Path.cwd().resolve()
candidates = [p for parent in (cwd, *cwd.parents)
              for p in (parent, parent / 'experiments' / 'decomposition')]
EXP = next(p for p in candidates if (p / 'results' / 'fft_reconstruction_summary.csv').exists())
R = EXP / 'results'
summary = pd.read_csv(R / 'fft_reconstruction_summary.csv')
by_seed = pd.read_csv(R / 'fft_reconstruction_by_seed.csv')
paired = pd.read_csv(R / 'fft_reconstruction_paired.csv')
identity = pd.read_csv(R / 'fft_reconstruction_identity.csv')
runs = pd.read_csv(R / 'fft_reconstruction_runs.csv')
topk = pd.read_csv(R / 'fft_reconstruction_topk.csv')
variants = ['original', 'amplitude', 'exact_frequency']
colors = dict(zip(variants, ['#64748b', '#f59e0b', '#2563eb']))
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})
""")
    md("""## Protocol and reproducibility

Context 512; horizons 96/128/256; seeds 0/1/2; 27 runs of 30 epochs each.
2048 training and 256 validation series from the benchmark's 45 original synthetic scenarios.
Test: 512 base series + 12 diagnostic cases × 32 series, absent from training and validation.
Different horizons use prefixes of the same series of length 768.

AdamW 0.001, batch size 32, gradient clipping 1. Loss is the **unchanged `benchmark.runner.loss`**:
Gaussian reconstruction NLL + forecast NLL + 0.01 KL, including its complex latent convention.
Initial weights, batch order, and random number streams are identical within each
seed/horizon pair. The input decomposition remains trainable and may diverge after training.
Validation runs every epoch (32 latent + observation draws); the best checkpoint
is selected by validation CRPS. Test uses 128 draws; mean error is computed from the mean latent forecast.

This is a new controlled experiment with a smaller training set and budget than `night` in the original benchmark.
Runtime, environment versions, and source and data hashes are recorded for each run.
The local experiment used the available Python/PyTorch environment, which differs from `uv.lock`.
""")
    prov = json.loads((root / "provenance.json").read_text())
    code(f"""provenance = {repr(prov)}
print('Python:', provenance['python'].split()[0], '| torch:', provenance['versions']['torch'])
print('Device:', provenance['device'], '| source/model hashes:', len(provenance['source_hashes']))
assert len(runs) == 27 and runs.epochs.eq(30).all()
assert runs.groupby(['horizon', 'seed']).initial_state_sha256.nunique().eq(1).all()
assert runs.parameter_count.nunique() == 1
print('27 runs, 30 epochs, identical paired initialization; parameters:', runs.parameter_count.iloc[0])
display(runs[['horizon', 'seed', 'variant', 'best_epoch', 'best_val_ncrps', 'seconds']])
""")
    md("""## Check without training

The true top-16 FFT coefficients are passed directly to the three reconstruction operations.
For periods that divide 512, `exact_frequency` should preserve the curve to numerical precision.
For non-integer periods, the FFT representation remains a limitation; correcting the remapping alone
does not eliminate spectral leakage or T-periodic continuation.
""")
    code("""display(identity[identity.period.isin([32, 64, 128, 256])].pivot_table(
    index=['horizon', 'period'], columns='variant', values='nrmse_horizon').round(4))
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
periods = [32., 64., 128., 256., 19.7, 37.1, 168.]
for ax, horizon in zip(axes, [96, 128, 256]):
    for offset, variant in enumerate(variants):
        values = identity[(identity.horizon == horizon) & (identity.variant == variant)].set_index('period')
        ax.bar(np.arange(len(periods)) + (offset-1)*.24, values.loc[periods, 'nrmse_horizon'],
               width=.23, label=variant, color=colors[variant])
    ax.set_xticks(np.arange(len(periods)), [str(p).removesuffix('.0') for p in periods], rotation=45)
    ax.set_title(f'Identity decoder, H={horizon}')
    ax.axhline(1, color='black', linestyle=':', linewidth=1)
    ax.set_xlabel('True period')
axes[0].set_ylabel('Horizon RMSE / true seasonal std')
axes[-1].legend(fontsize=8)
fig.tight_layout()
plt.show()
""")
    md("""## Forecast after training

Base test from 45 scenarios. Bars show the mean across three training seeds;
error bars show the standard deviation across seeds, not confidence intervals.
Lower CRPS is better; nominal coverage is 90%.
""")
    code("""base = summary[summary.case == 'base']
display(base[['horizon', 'variant', 'ncrps_mean', 'ncrps_std', 'nrmse_mean',
              'coverage90_mean', 'seasonal_hor_nrmse_mean', 'seasonal_gain_mean']].round(5))
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for ax, horizon in zip(axes, [96, 128, 256]):
    table = base[base.horizon == horizon].set_index('variant').loc[variants]
    ax.bar(range(3), table.ncrps_mean, yerr=table.ncrps_std, capsize=4,
           color=[colors[v] for v in variants])
    ax.set_xticks(range(3), ['original', 'amplitude', 'exact freq.'], rotation=20)
    ax.set_title(f'Horizon {horizon}')
axes[0].set_ylabel('Normalized CRPS (lower is better)')
fig.tight_layout()
plt.show()
""")
    md("""## Paired comparisons and diagnostic cases

Δ = new variant − original: a negative value indicates improvement.
95% intervals use 2000 paired bootstrap resamples of test series **within each seed**.
They describe variation across test series, not uncertainty across new training seeds.
The diagnostic checks are exploratory; no correction for multiple comparisons was applied.
""")
    code("""paired_base = paired[(paired.case == 'base') & (paired.metric == 'ncrps')]
display(paired_base[['horizon', 'seed', 'variant', 'delta', 'ci_low', 'ci_high', 'improvement_pct']].round(6))
effects = paired[paired.metric == 'ncrps'].groupby(['horizon', 'variant', 'case']).delta.mean()
cases = ['base', 'period_32', 'period_64', 'period_128', 'period_256', 'period_19_7',
         'period_37_1', 'period_168', 'dual_24_168', 'dual_17_3_96',
         'amplitude_growth', 'negative_noise_only', 'negative_trend_noise']
fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
limit = max(abs(effects.min()), abs(effects.max()), 1e-6)
for ax, variant in zip(axes, ['amplitude', 'exact_frequency']):
    matrix = np.array([[effects.loc[(h, variant, c)] for h in [96, 128, 256]] for c in cases])
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
    for i in range(len(cases)):
        for j in range(3):
            ax.text(j, i, f'{matrix[i,j]:+.3f}', ha='center', va='center', fontsize=8,
                    color='white' if abs(matrix[i,j]) > .65*limit else 'black')
    ax.set_xticks(range(3), [96, 128, 256]); ax.set_yticks(range(len(cases)), cases)
    ax.set_title(variant); ax.set_xlabel('Horizon')
fig.colorbar(im, ax=axes, label='CRPS delta vs original (negative is better)', shrink=.8)
plt.show()
""")
    md("""## Seasonal branch usage

`seasonal_gain` is the projection of the decoded seasonal component onto the true seasonal component in the context
(expected to be close to 1 for accurate recovery). Gain is undefined for series without seasonality.
`input_leak_abs` is the absolute projection of the input trend error onto the true seasonal component.
These coefficients are not energy fractions. Component error accounts for amplitude, phase, and shape.
We also measure the fraction of true seasonal energy in the FFT bins selected by
the seasonal encoder from the standard MA residual. Ground truth is used only for this diagnostic,
not as model input. A small value means that the relevant frequencies were missed by top-K selection.
""")
    code("""display(base[['horizon', 'variant', 'input_leak_abs_mean', 'input_seasonal_nrmse_mean',
              'seasonal_gain_mean', 'seasonal_share_mean', 'seasonal_hor_nrmse_mean']].round(5))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, metric, label in zip(axes, ['seasonal_gain', 'seasonal_hor_nrmse'],
                            ['Decoded seasonal projection gain', 'Seasonal horizon NRMSE']):
    for offset, variant in enumerate(variants):
        table = base[base.variant == variant].set_index('horizon').loc[[96, 128, 256]]
        ax.bar(np.arange(3)+(offset-1)*.24, table[metric+'_mean'], width=.23,
               yerr=table[metric+'_std'], capsize=3, label=variant, color=colors[variant])
    ax.set_xticks(range(3), [96, 128, 256]); ax.set_xlabel('Horizon'); ax.set_ylabel(label)
axes[0].axhline(1, color='black', linestyle=':', linewidth=1)
axes[-1].legend(fontsize=8)
fig.tight_layout()
plt.show()
""")
    code("""retention = topk.groupby(['horizon', 'variant', 'case']).topk_true_seasonal_energy.mean()
display(retention.unstack('case')[['base', 'period_32', 'period_64', 'period_128', 'period_256']].round(4))
trend_leak = topk.groupby(['horizon', 'variant', 'case']).decoded_trend_seasonal_gain.mean()
print('Projection of decoded trend error onto true seasonal (ideal: zero):')
display(trend_leak.unstack('case')[['base', 'period_32', 'period_64', 'period_128', 'period_256']].round(4))
""")
    md("""## Example after training: period 64, seed 0, horizon 96

The first series of a diagnostic case selected in advance, rather than an example with the largest improvement.
Forecast and component means use 128 latent draws from the best validation checkpoint.
Checkpoint files and full predictions are stored in the Git-ignored runs folder.
""")
    code(f"""run_root = Path({str(root)!r})
# When reproducing elsewhere, run train.py with the default run id, or set DECOVAE_RUNS.
if not run_root.exists():
    import os
    run_root = Path(os.environ.get('DECOVAE_RUNS', EXP / 'runs')) / 'fft_reconstruction_full'
if (run_root / 'test.npz').exists():
    raw = dict(np.load(run_root / 'test.npz'))
    row = int(np.flatnonzero(raw['case'] == 'period_64')[0])
    horizon = 96
    t = np.arange(512+horizon)
    gt_seas = raw['seasonal'][row, :512+horizon] / raw['scale'][row, 0]
    gt_trend = (raw['trend'][row, :512+horizon] - raw['mean'][row, 0]) / raw['scale'][row, 0]
    mean = (raw['trend'][row, :512+horizon] + raw['seasonal'][row, :512+horizon]
            - raw['mean'][row, 0]) / raw['scale'][row, 0]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    axes[0].plot(t, gt_seas, color='black', label='true seasonal', linewidth=1.5)
    axes[1].plot(t, gt_trend, color='black', label='true trend', linewidth=1.5)
    axes[2].plot(t[512:], mean[512:], color='black', label='true conditional mean', linewidth=1.5)
    for variant in variants:
        predictions = dict(np.load(run_root / f'h96_s0_{{variant}}' / 'predictions.npz'))
        axes[0].plot(t, predictions['seasonal'][row], label=variant, color=colors[variant])
        axes[1].plot(t, predictions['trend'][row], label=variant, color=colors[variant])
        axes[2].plot(t[512:], predictions['mean'][row], label=variant, color=colors[variant])
    axes[0].axvline(512, color='gray', linestyle=':')
    axes[1].axvline(512, color='gray', linestyle=':')
    axes[0].set_ylabel('Seasonal component'); axes[1].set_ylabel('Trend component')
    axes[2].set_ylabel('Forecast mean')
    for ax in axes: ax.legend(fontsize=8)
    axes[2].set_xlabel('Time'); fig.tight_layout(); plt.show()
else:
    print('Raw run outputs absent; figures above and saved notebook outputs remain available.')
""")
    conclusions = (Path(__file__).parent / "FINDINGS.md").read_text()
    md(conclusions)
    return nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "name": "python3", "language": "python"},
        "language_info": {"name": "python", "version": sys.version.split()[0]}})


def write_findings(root, summary, paired, statuses):
    provenance = json.loads((root / "provenance.json").read_text())
    base = summary[summary.case == "base"]
    text = ["# FFT-only reconstruction ablation", "",
            "Completed: 27 runs, 3 variants × 3 horizons (96/128/256) × 3 training seeds (0/1/2), 30 epochs each.",
            "Native DecoVAEFM; only `_reconstruct_time_from_freq_tokens` differs. Original MA, learned encoders,",
            "trend/seasonal/variance heads, noise latent and unchanged `benchmark.runner.loss` are shared.",
            "All pairs have identical initial parameter hashes and batch orders. The learned MA weights remain trainable.",
            "", "## Base test: mean ± SD over training seeds", "",
            "CRPS and NRMSE are in context-standardized units; lower is better. SD is not a confidence interval.", "",
            "| H | Variant | CRPS | Forecast NRMSE | Coverage90 | Seasonal horizon NRMSE | Seasonal gain (context) |",
            "|---|---|---|---|---|---|---|"]
    for horizon in (96, 128, 256):
        for variant in ("original", "amplitude", "exact_frequency"):
            r = base[(base.horizon == horizon) & (base.variant == variant)].iloc[0]
            text.append(f"| {horizon} | {variant} | {r.ncrps_mean:.5f} ± {r.ncrps_std:.5f} | "
                        f"{r.nrmse_mean:.5f} | {r.coverage90_mean:.4f} | {r.seasonal_hor_nrmse_mean:.5f} | "
                        f"{r.seasonal_gain_mean:.5f} |")
    text += ["", "## Paired forecast CRPS differences", "",
             "Delta = corrected − original. Negative is better. Intervals are paired row-bootstrap (2000 resamples)",
             "inside a training seed; they do not quantify uncertainty over future training seeds.", "",
             "| H | Variant | Mean delta over seeds | Improved seeds | Seeds with CI entirely below zero |",
             "|---|---|---|---|---|"]
    deltas = paired[(paired.case == "base") & (paired.metric == "ncrps")]
    for (horizon, variant), d in deltas.groupby(["horizon", "variant"]):
        text.append(f"| {horizon} | {variant} | {d.delta.mean():+.6f} | {(d.delta < 0).sum()}/3 | "
                    f"{(d.ci_high < 0).sum()}/3 |")
    text += ["", "## Observed results", ""]
    for variant in ("amplitude", "exact_frequency"):
        relevant = deltas[deltas.variant == variant]
        strong = [str(h) for h, d in relevant.groupby("horizon") if (d.ci_high < 0).sum() >= 2]
        text.append(f"- `{variant}`: base-test improvement with a bootstrap interval entirely below zero in "
                    f"at least 2/3 seeds at horizons: {', '.join(strong) if strong else 'none'}.")
        cases = paired[(paired.variant == variant) & (paired.metric == "ncrps") & (paired.case != "base")]
        worse = [(h, c) for (h, c), d in cases.groupby(["horizon", "case"]) if (d.delta > 0).all()]
        text.append(f"  Diagnostic case × horizon pairs worse in all three seeds: {len(worse)} / 36 "
                    "(exploratory signs, not multiplicity-corrected significance).")
    gain_min, gain_max = base.seasonal_gain_mean.min(), base.seasonal_gain_mean.max()
    text.append(f"- Average decoded seasonal projection gains on the base test range from {gain_min:.4f} to "
                f"{gain_max:.4f}; an exact component match would have gain 1.")
    if gain_max < .5:
        text.append("  None of these decoder-only variants restores full seasonal component recovery on this training profile.")
    representation = pd.read_csv(EXPERIMENTS / "results/fft_reconstruction_topk.csv")
    for case in ("period_64", "period_128", "period_256"):
        data = representation[representation.case == case]
        text.append(f"- `{case}`: mean fraction of true seasonal FFT variance retained in the encoder's top-K "
                    f"bins is {data.topk_true_seasonal_energy.mean():.4f}; decoded trend's seasonal projection gain "
                    f"is {data.decoded_trend_seasonal_gain.mean():.4f} (ideal zero), averaged over this grid.")
    reparameterization = root / "amplitude_reparameterization_check.json"
    if reparameterization.exists():
        check = json.loads(reparameterization.read_text())
        max_error = max(check["max_abs_errors"].values())
        text.append(f"- In the trained original H96/seed0 checkpoint, scaling the final complex-linear seasonal head "
                    f"by 512/608 compensates amplitude-only reconstruction: maximum output difference {max_error:.3g} "
                    "on eight fixed test contexts with paired latent draws.")
    text += ["", "## Interpretation and scope", "",
             "- The identity check confirms the amplitude convention and bin-rounding defects without training.",
             "  Exact-frequency evaluation reconstructs periods dividing T=512 up to numerical precision.",
             "- Exact-frequency evaluation is a T-periodic extension of the same top-K bins, not a true-period estimator.",
             "  Off-grid seasonality can still extrapolate poorly. A decoder-only fix does not remove this limitation.",
             "  In particular, the base positive periods 12/24/48/96 do not divide 512. The exact-k/T decoder forces",
             "  seasonal(t+512)=seasonal(t), which the true seasonal does not satisfy; the trend head can compensate",
             "  in the total mean, but this does not restore true component recovery.",
             "- For a fixed horizon, amplitude-only correction is a constant rescaling which the learned seasonal head",
             "  can absorb. The identity error is real, but a forecast improvement after training is not guaranteed.",
             "- Trained performance and component recovery must be assessed separately using the table above and diagnostics.",
             "  An operator defect does not by itself establish that correcting it fixes learned decomposition.",
             "- Per-case diagnostics, paired intervals and three horizon segments are in results/fft_reconstruction_*.csv.",
             "  Diagnostic comparisons are exploratory and have no multiple-comparison correction.",
             "", "## Reproduction and limitations", "",
             "2048 train / 256 validation / 512 base test + 12 × 32 held-out diagnostic series. Linear context-only",
             "normalization, AdamW 0.001, batch32, clip1. Best validation-CRPS checkpoint from a fixed 30-epoch budget;",
             "32 validation and 128 test latent + observation draws. No test metric is used to select checkpoints.",
             "Same underlying paths are sliced for all horizons. Series seeds: 719 / 100719 / 200719.",
             "Synthetic data only; three training seeds; this is not the larger existing `night` benchmark.",
             "Each horizon is trained separately. Generalization of one trained model to an unseen horizon is not tested.",
             f"Local runtime: Python {sys.version.split()[0]}, torch {provenance['versions']['torch']}, CPU; differs from uv.lock.",
             "Provenance, data hashes, source snapshots, checkpoints, histories and per-series outputs are under",
             f"`runs/{root.name}/` (Git-ignored). Summaries and the executed notebook are tracked artifacts."]
    (Path(__file__).parent / "FINDINGS.md").write_text("\n".join(text) + "\n")


def execute_notebook(notebook, path):
    """Execute locally without a Jupyter socket; retain all figures in the notebook."""
    from IPython.terminal.interactiveshell import TerminalInteractiveShell
    from IPython.utils.capture import capture_output
    shell = TerminalInteractiveShell.instance()
    shell.run_line_magic("matplotlib", "inline")
    shell.display_formatter.active_types = ["text/plain", "text/html", "image/png"]
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=True)
        error = result.error_before_exec or result.error_in_exec
        if error:
            raise error
        cell.execution_count = count
        cell.outputs = []
        if captured.stdout:
            cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
        if captured.stderr:
            cell.outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
        for output in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output("display_data", data=output.data, metadata=output.metadata))
        print(f"Notebook cell {count}: executed", flush=True)
    notebook.metadata["execution"] = {"method": "IPython in-process; no Jupyter port required"}
    nbformat.validate(notebook)
    nbformat.write(notebook, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="fft_reconstruction_full")
    parser.add_argument("--no-execute", action="store_true")
    args = parser.parse_args()
    root = RUNS_DIR / args.run_id
    status = json.loads((root / "status.json").read_text())
    if status["status"] != "complete" or status["runs"] != 27 or status["config"]["profile"] != "full":
        raise ValueError("Publish summaries only for the completed 27-run full grid")
    rows = pd.read_csv(root / "all_metrics.csv")
    runs = [json.loads(p.read_text()) for p in sorted(root.glob("h*_s*_*/status.json"))]
    assert len(runs) == 27 and all(r["epochs"] == 30 for r in runs)
    for h in status["config"]["horizons"]:
        for seed in status["config"]["seeds"]:
            group = [r for r in runs if r["horizon"] == h and r["seed"] == seed]
            assert len({r["initial_state_sha256"] for r in group}) == 1
            assert len({tuple(r["batch_order_sha256"]) for r in group}) == 1
    metrics = list(rows.select_dtypes("number").columns.difference(["series_id", "scenario", "seed", "horizon"]))
    by_seed = rows.groupby(["horizon", "seed", "variant", "case"], as_index=False)[metrics].mean()
    summary = by_seed.groupby(["horizon", "variant", "case"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    paired = paired_summary(rows)
    run_table = pd.DataFrame([{k: v for k, v in r.items() if k not in (
        "batch_order_sha256", "initial_decomposition_weights", "final_decomposition_weights")}
                              for r in runs])
    results = EXPERIMENTS / "results"
    results.mkdir(exist_ok=True)
    for name, data in (("by_seed", by_seed), ("summary", summary), ("paired", paired), ("runs", run_table)):
        data.to_csv(results / f"fft_reconstruction_{name}.csv", index=False)
    identity_table().to_csv(results / "fft_reconstruction_identity.csv", index=False)
    topk_signal_retention(root, runs).to_csv(results / "fft_reconstruction_topk.csv", index=False)
    write_findings(root, summary, paired, runs)
    notebook = build_notebook(root, summary, paired)
    target = EXPERIMENTS / "fft_reconstruction_summary.ipynb"
    if args.no_execute:
        nbformat.write(notebook, target)
    else:
        execute_notebook(notebook, target)
    artifacts = list(results.glob("fft_reconstruction_*.csv")) + [target, Path(__file__).parent / "FINDINGS.md"]
    import hashlib
    writeup = dict(report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   training_provenance_sha256=hashlib.sha256((root / "provenance.json").read_bytes()).hexdigest(),
                   artifacts={str(p.relative_to(EXPERIMENTS)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in artifacts},
                   ipython_version=__import__("IPython").__version__, nbformat_version=nbformat.__version__)
    (root / "report_provenance.json").write_text(json.dumps(writeup, indent=2))
    print(target)


if __name__ == "__main__":
    main()
