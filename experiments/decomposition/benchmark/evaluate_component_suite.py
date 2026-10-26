"""Measure input-component recovery on the extended synthetic diagnostic suite."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from adapters import Arm, algorithm
from fourier_arm import FOURIER_ARMS,make_model,attach
from data import L, T, ROOT, PERIOD_DECOMP_DIR, normalize
from metrics import components
from runner import OUT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', required=True)
    parser.add_argument('--runs-csv', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--no-correction', action='store_true')
    args = parser.parse_args()

    suite = Path(args.suite)
    raw = {k: v for k, v in np.load(suite / 'test.npz').items()}
    z, mean, scale = normalize(raw['x'])
    runs = pd.read_csv(args.runs_csv).model_run.drop_duplicates().tolist()
    torch.set_num_threads(1)

    period_cache = None; fourier = None
    if any(json.loads((OUT / run / 'provenance.json').read_text())['config']['arm'] in FOURIER_ARMS for run in runs):
        fourier = attach(dict(z=z), suite, 'suite')
    if any(json.loads((OUT / run / 'provenance.json').read_text())['config']['arm'] in 'CD' for run in runs):
        digest = hashlib.sha256(
            (PERIOD_DECOMP_DIR / 'period_decomp.py').read_bytes()
            + (Path(__file__).parent / 'adapters.py').read_bytes()
        ).hexdigest()[:12]
        cache = suite / f'period_{digest}.npz'
        if cache.exists():
            period_cache = np.load(cache)['components']
        else:
            period_cache = np.stack([
                np.stack([algorithm(row[:T], 'period')[0][key] for key in ('trend', 'seasonal')])
                for row in z
            ]).astype('float32')
            np.savez_compressed(cache, components=period_cache)

    rows = []
    for position, run in enumerate(runs, 1):
        model_dir = OUT / run
        provenance = json.loads((model_dir / 'provenance.json').read_text())
        config = provenance['config']
        arm = config['arm']
        model = make_model(arm)
        checkpoint = torch.load(model_dir / 'best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        if args.no_correction: model.use_correction = False
        model.reparameterize = lambda mu, logvar: mu
        trend_parts, seasonal_parts = [], []
        decoded_trend_parts, decoded_seasonal_parts, forecast_parts = [], [], []
        for start in range(0, len(z), args.batch_size):
            stop = min(start + args.batch_size, len(z))
            x = torch.from_numpy(z[start:stop, :T].astype('float32'))
            if arm in FOURIER_ARMS:
                model.components = (torch.from_numpy(fourier['components'][start:stop, 0]), torch.from_numpy(fourier['components'][start:stop, 1]))
                model.fourier = (torch.from_numpy(fourier['periods'][start:stop]), torch.from_numpy(fourier['coefs'][start:stop]))
            elif arm in 'CD':
                model.components = (
                    torch.from_numpy(period_cache[start:stop, 0]),
                    torch.from_numpy(period_cache[start:stop, 1]),
                )
            with torch.no_grad():
                result = model(x)
            trend_parts.append(result['target_trend'].cpu().numpy())
            seasonal_parts.append(result['target_seasonal'].cpu().numpy())
            decoded_trend_parts.append(result['decoded_trend'].cpu().numpy())
            decoded_seasonal_parts.append(result['decoded_seasonal'].cpu().numpy())
            forecast_parts.append(result['forecast_mu'].cpu().numpy())
        estimated_trend = np.concatenate(trend_parts) * scale + mean
        estimated_seasonal = np.concatenate(seasonal_parts) * scale
        decoded_trend = np.concatenate(decoded_trend_parts) * scale + mean
        decoded_seasonal = np.concatenate(decoded_seasonal_parts) * scale
        forecast = np.concatenate(forecast_parts) * scale + mean

        for i in range(len(raw['x'])):
            params = json.loads(str(raw['params_json'][i]))
            has_trend = bool(np.any(np.abs(raw['trend'][i]) > 1e-12))
            sigma = float(np.sqrt(np.mean(raw['variance'][i, :T])))
            record = dict(
                x=raw['x'][i], trend=raw['trend'][i], seasonal=raw['seasonal'][i],
                noise=raw['noise'][i], params=params,
                scenario=('linear' if has_trend else 'none', 0, sigma, 'diagnostic'),
            )
            estimate = {
                'trend': estimated_trend[i],
                'seasonal': estimated_seasonal[i],
            }
            estimate['noise'] = raw['x'][i, :T] - estimate['trend'] - estimate['seasonal']
            decoded = {
                'trend': decoded_trend[i, :T],
                'seasonal': decoded_seasonal[i, :T],
            }
            decoded['noise'] = raw['x'][i, :T] - decoded['trend'] - decoded['seasonal']
            row = dict(model_run=run, arm=arm + ('0' if args.no_correction else ''), seed=int(config['seed']),
                       case=str(raw['case'][i]), series_id=i)
            row.update({'input_' + k: v for k, v in components(record, estimate).items()})
            row.update({'decoded_' + k: v for k, v in components(record, decoded).items()})
            row['posterior_mean_forecast_mse'] = float(np.mean((forecast[i] - raw['forecast_mean'][i]) ** 2))
            rows.append(row)
        print(f'component suite {position}/{len(runs)}: {run}', flush=True)

    frame = pd.DataFrame(rows)
    output = OUT / args.out
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / 'component_rows.csv', index=False)

    metrics = [c for c in frame.columns if c.endswith('_nrmse') or c.endswith('_residual_seasonal_energy')
               or c.endswith('_false_seasonal_rms') or c.endswith('_false_trend_rms')
               or c == 'posterior_mean_forecast_mse']
    rng = np.random.default_rng(20261009)
    summary = []
    for (run, case), group in frame.groupby(['model_run', 'case'], sort=True):
        item = dict(model_run=run, arm=group.arm.iloc[0], seed=int(group.seed.iloc[0]), case=case, n=len(group))
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            item[metric + '_mean'] = float(values.mean()) if len(values) else np.nan
            if len(values):
                draws = rng.choice(values, (2000, len(values)), replace=True).mean(axis=1)
                low, high = np.quantile(draws, [0.025, 0.975])
                item[metric + '_ci_low'] = float(low)
                item[metric + '_ci_high'] = float(high)
            else:
                item[metric + '_ci_low'] = np.nan
                item[metric + '_ci_high'] = np.nan
        summary.append(item)
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(output / 'component_summary.csv', index=False)
    (output / 'provenance.json').write_text(json.dumps({
        'suite': str(suite), 'runs': runs, 'estimand': 'input component recovery before decoder',
        'decoded_components': 'deterministic decoder at posterior means',
        'rows_per_case': int(json.loads((suite / 'config.json').read_text())['n_per_case']),
        'bootstrap_replicates': 2000, 'device': 'cpu',
    }, indent=2) + '\n')
    selected = ['negative_noise_only', 'negative_season_noise', 'period_37_1',
                'dual_17_3_96_ratio_1', 'dual_17_3_96_ratio_3']
    view = summary_frame[summary_frame.case.isin(selected)].groupby(['case', 'arm']).mean(numeric_only=True).reset_index()
    columns = ['input_seasonal_nrmse_mean', 'decoded_seasonal_nrmse_mean',
               'input_noise_nrmse_mean', 'decoded_noise_nrmse_mean',
               'posterior_mean_forecast_mse_mean']
    display = view[['case', 'arm'] + columns].rename(columns={
        'case': 'Case', 'arm': 'Arm', 'input_seasonal_nrmse_mean': 'Season input NRMSE',
        'decoded_seasonal_nrmse_mean': 'Season decoded NRMSE',
        'input_noise_nrmse_mean': 'Noise input NRMSE',
        'decoded_noise_nrmse_mean': 'Noise decoded NRMSE',
        'posterior_mean_forecast_mse_mean': 'Posterior-mean forecast MSE',
    })
    missing_training = sorted({f'{a}_s{s}' for s in range(3) for a in 'ABCD'} -
                              {f"{json.loads((OUT / run / 'provenance.json').read_text())['config']['arm']}_s{int(json.loads((OUT / run / 'provenance.json').read_text())['config']['seed'])}" for run in runs})
    report = [
        '# Input decomposition and decoder diagnostics', '',
        f'{len(runs)} available checkpoints were evaluated on {len(np.unique(raw["case"]))} cases × {json.loads((suite / "config.json").read_text())["n_per_case"]} rows. Missing training cells: {", ".join(missing_training) if missing_training else "none"}.', '',
        'Input metrics compare encoded trend and season components with synthetic truth before decoding. Decoded metrics use a deterministic forward pass with latent posterior means. Forecast MSE is the resulting deterministic forecast mean against the known conditional mean; it is not a CRPS estimate. Intervals in `component_summary.csv` resample rows within a case and seed, not training seeds.', '',
        '## Selected cases', '', display.to_string(index=False, float_format=lambda value: f'{value:.3f}'), '',
        'See `component_rows.csv` for per-row values and `component_summary.csv` for all case/seed intervals.',
    ]
    (output / 'report.md').write_text('\n'.join(report) + '\n')
    print(f'Wrote {len(frame)} series rows and {len(summary)} model-case summaries to {output}', flush=True)


if __name__ == '__main__':
    main()
