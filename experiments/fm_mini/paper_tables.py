"""Tables of the report, rebuilt from results/ only (no data, checkpoints or GPU needed).

Skill = geometric mean over sources of (mean nCRPS of the model / mean nCRPS of seasonal-naive), lower is better.
Means and sd are over training seeds. Ratio CIs: paired cluster bootstrap over series within source on seed-averaged
window values (2000 draws), the ratio being exp(mean over sources of log(sum_model / sum_reference)).
Ablation ratios pair runs by seed and report the median over seeds.

Usage: python paper_tables.py            # prints all tables and writes results/tables.md
Other code (the overview notebook) imports the functions below.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent/'results'
SPLITS = {'test_ood': 'OOD (held-out sources)', 'test_id': 'ID (familiar sources)'}

# Report name -> run family (run ids are <family>_s<seed>).
MAIN = {'NLL': ('srv4_N6r02', 'srv5_Dcomb'), 'Mix': ('srv10_N6mix', 'srv8_Dmix'), 'MixC': ('srv11_N6mc1', 'srv10_Dmc1')}
EXTRA = {'DV-FM (NLL)': 'srv9_Ar02', 'TF-6L-Mix + 50% KernelSynth': 'srv10_N6mix_ks50'}
CAPACITY = {'TF-2L': ('srv10_N2r1', 'srv10_N2r02'), 'DV-FM': ('srv2b3_A_f0', 'srv9_Ar02'), 'TF-6L': ('srv2b3_N6_f0', 'srv4_N6r02')}
UNCERTAINTY = {'TF-6L': 'srv4_N6r02', 'TF-6L-MixC': 'srv11_N6mc1', 'PD': 'srv5_Dcomb', 'PD-Mix': 'srv8_Dmix', 'PD-CRPS': 'srv8_Dcrps',
               'PD-MixC': 'srv10_Dmc1', 'PD-CVAE-S': 'srv7_Dzsh', 'PD-CVAE-G': 'srv7_Dzgh'}
KERNELSYNTH = {'TF-6L': ['srv4_N6r02', 'srv7_N6r02_ks25', 'srv7_N6r02_ks50', 'srv7_N6r02_ks100'],
               'PD': ['srv5_Dcomb', 'srv7_Dcomb_ks25', 'srv7_Dcomb_ks50', 'srv7_Dcomb_ks100']}
SCALE = {'d64, 1x': ('srv11_N6mc1', 'srv10_Dmc1'), 'd128, 16x': ('srv11_N6mc1_d128x16', 'srv11_Dmc1_d128x16')}
LAB = {'DV-FM': 'lab_A', 'TF-6L': 'lab_N6r02', 'PD': 'lab_Dcomb', 'PD-MixC': 'lab_Dmc1'}
# (change, reference, numerator families, denominator families, score column). Families scored only by train.py's
# evaluation (no eval_space output) use that evaluation; asinh rows use the guarded CRPS (samples clipped to the context
# range +- 3 ranges), because a Gaussian in asinh space has exponential tails in x and a few windows dominate the mean.
ABLATIONS = [
    ('input without seasonal part, no level anchor', 'reference', 'srv4_Dd', 'srv4_Df', 'ncrps'),
    ('input without seasonal part + level anchor', 'reference', 'srv4_Dda', 'srv4_Df', 'ncrps'),
    ('reliability features in the variance', 'reference', 'srv4_Dfs', 'srv4_Df', 'ncrps'),
    ('all of the above + reconstruction weight 0.2 = PD', 'reference', 'srv5_Dcomb', 'srv4_Df', 'ncrps'),
    ('reconstruction weight 0.2 instead of 1', 'same model', 'srv4_Cfr02', 'srv4_Cf', 'ncrps'),
    ('trained without the learned Fourier-coefficient correction', 'with it', 'srv2b1_Cf0', ('srv1_Cf', 'srv2b1_Cf'), 'ncrps'),
    ('forecast-only correction S1', 'computed seasonal', 'srv3_Cs1', 'srv3_Cf0', 'ncrps'),
    ('forecast-only correction S2', 'computed seasonal', 'srv3_Cs2', 'srv3_Cf0', 'ncrps'),
    ('S1 + S2 on top of PD-MixC', 'PD-MixC', 'srv11_Dmc1s', 'srv10_Dmc1', 'ncrps'),
    ('noise stream, batch 32', 'without', ('srv1_Cf', 'srv2b1_Cf'), ('srv1_Df', 'srv2b1_Df'), 'ncrps'),
    ('noise stream, batch 256', 'without', 'srv4_Cf', 'srv4_Df', 'ncrps'),
    ('smoothness penalty on the non-seasonal part (HP, lambda 1e-3)', 'PD', 'srv9_DhpS', 'srv5_Dcomb', 'ncrps'),
    ('smoothness penalty on the non-seasonal part (HP, lambda 1e-1)', 'PD', 'srv9_DhpM', 'srv5_Dcomb', 'ncrps'),
    ('hard HP smoothing, cut-off 1.5 x longest period', 'PD', 'srv9_DhpP', 'srv5_Dcomb', 'ncrps'),
    ('hard HP smoothing, cut-off T/2', 'PD', 'srv9_DhpT', 'srv5_Dcomb', 'ncrps'),
    ('asinh output space, capped variance', 'PD', 'srv9_DcombCAstd', 'srv5_Dcomb', 'ncrps_g'),
    ('asinh output space, capped variance', 'PD-Mix', 'srv10_DmixCAstd', 'srv8_Dmix', 'ncrps_g'),
]


@lru_cache(None)
def runs():
    return pd.read_csv(RESULTS/'runs.csv')


@lru_cache(None)
def scores():
    """Window scores of both evaluations; eval = 'space' (eval_space.py) or 'train' (train.py, older runs)."""
    a = pd.read_parquet(RESULTS/'window_scores.parquet').assign(eval='space')
    b = pd.read_parquet(RESULTS/'train_eval_scores.parquet').assign(eval='train')
    a['ncrps_g'] = a.ncrps_g.fillna(a.ncrps)  # stored only where the guard changed the value
    d = pd.concat([a, b], ignore_index=True)
    for c in ('run', 'split', 'dataset', 'eval'): d[c] = d[c].astype(str)
    return d


@lru_cache(None)
def calibration():
    d = pd.read_parquet(RESULTS/'calibration.parquet')
    for c in ('run', 'split', 'dataset'): d[c] = d[c].astype(str)
    return d


@lru_cache(None)
def seasonal_naive(split, protocol='main'):
    b = pd.read_parquet(RESULTS/'baselines.parquet')
    b = b[(b.protocol.astype(str) == protocol) & (b.method.astype(str) == 'seasonal_naive') & (b.split.astype(str) == split)]
    return b.groupby(b.dataset.astype(str)).ncrps.mean()


def baseline_skill(method, split, protocol='main'):
    b = pd.read_parquet(RESULTS/'baselines.parquet')
    b = b[(b.protocol.astype(str) == protocol) & (b.method.astype(str) == method) & (b.split.astype(str) == split)]
    if b.empty: return np.nan
    per = b.groupby(b.dataset.astype(str)).ncrps.mean(); sn = seasonal_naive(split, protocol)
    return float(np.exp(np.log(per/sn.reindex(per.index)).mean()))


def members(family):
    """Run ids of one family, or of several families given as a tuple (e.g. seeds split across two series)."""
    fams = (family,) if isinstance(family, str) else tuple(family)
    r = runs(); return sorted(r[r.family.isin(fams)].run)


@lru_cache(None)
def _by_run():
    return {k: g.sort_values('window') for k, g in scores().groupby(['run', 'split'])}


def windows(run, split):
    return _by_run().get((run, split), scores().iloc[:0])


def run_skill(run, split, col='ncrps', protocol='main'):
    w = windows(run, split)
    if w.empty: return np.nan
    per = w.groupby('dataset')[col].mean(); sn = seasonal_naive(split, protocol)
    return float(np.exp(np.log(per/sn.reindex(per.index)).mean()))


def family_skill(family, split, col='ncrps'):
    v = np.array([run_skill(r, split, col) for r in members(family)]); v = v[np.isfinite(v)]
    return (v.mean(), v.std(ddof=1) if len(v) > 1 else np.nan, len(v)) if len(v) else (np.nan, np.nan, 0)


def per_source_skill(family, split):
    per = pd.concat([windows(r, split).groupby('dataset').ncrps.mean() for r in members(family)], axis=1).mean(1)
    return per/seasonal_naive(split).reindex(per.index)


def boot_ratio(nm, nb, ds, sid, n=2000, seed=991):
    """Skill ratio method/base = exp(mean_d log(sum_m / sum_b)), cluster bootstrap over series within source."""
    rng = np.random.default_rng(seed); logs = []; point = []; groups = []
    for d in np.unique(ds):
        s = ds == d; ids = sid[s]; u = np.unique(ids)
        cm = np.array([nm[s][ids == i].sum() for i in u]); cb = np.array([nb[s][ids == i].sum() for i in u])
        groups.append((cm, cb)); point.append(np.log(cm.sum()/cb.sum()))
    for _ in range(n):
        acc = []
        for cm, cb in groups:
            j = rng.integers(0, len(cm), len(cm)); acc.append(np.log(cm[j].sum()/cb[j].sum()))
        logs.append(np.mean(acc))
    return float(np.exp(np.mean(point))), np.exp(np.quantile(logs, [.025, .975]))


def seed_mean(family, split, col='ncrps'):
    ws = [windows(r, split) for r in members(family)]; ws = [w for w in ws if not w.empty]
    if not ws or any(len(w) != len(ws[0]) for w in ws): return None
    return np.mean([w[col].values for w in ws], 0), ws[0].dataset.values, ws[0].series_id.values


def ratio_ci(num, den, split, col='ncrps'):
    a, b = seed_mean(num, split, col), seed_mean(den, split, col)
    if a is None or b is None or len(a[0]) != len(b[0]): return np.nan, (np.nan, np.nan)
    return boot_ratio(a[0], b[0], a[1], a[2])


def ratio_by_seed(num, den, split, col='ncrps'):
    """Median over seeds of the paired skill ratio (same training seed), and the number of pairs."""
    seeds = lambda f: {int(r.rsplit('_s', 1)[1]): r for r in members(f) if r.rsplit('_s', 1)[1].isdigit()}
    a, b = seeds(num), seeds(den); common = sorted(set(a) & set(b))
    v = [run_skill(a[s], split, col)/run_skill(b[s], split, col) for s in common]; v = [x for x in v if np.isfinite(x)]
    return (float(np.median(v)), len(v)) if v else (np.nan, 0)


def uncertainty_row(family, split):
    rs = members(family); cal = calibration()
    rows = []
    for r in rs:
        c = cal[(cal.run == r) & (cal.split == split)]
        if c.empty: continue
        x = windows(r, split).groupby('dataset')[['nes', 'nvs', 'latent_share']].mean().mean()
        rows.append(dict(skill=run_skill(r, split), **c[['pce', 'cce', 'c50', 'c80', 'c98', 'up95']].mean(), **x))
    return pd.DataFrame(rows).mean() if rows else None


def lab_table():
    lab = pd.read_parquet(RESULTS/'lab_scores.parquet'); lab['run'] = lab.run.astype(str); rows = []
    for name, fam in LAB.items():
        d = lab[lab.run.str.match(rf'{fam}_s\d+$')]
        seen, new = d[d.split == 'test_id'], d[d.split == 'test_ood']
        rows.append(dict(model=name, seeds=d.run.nunique(),
                         share_P128=seen[seen.P == 128].lw_to_seasonal.mean() if seen.lw_to_seasonal.notna().any() else np.nan,
                         share_P192plus=seen[seen.P >= 192].lw_to_seasonal.mean() if seen.lw_to_seasonal.notna().any() else np.nan,
                         seen_crps_true=seen.crps_ratio.mean(), seen_c80=seen.c80.mean(), new_crps_true=new.crps_ratio.mean(), new_c80=new.c80.mean()))
    return pd.DataFrame(rows).set_index('model').astype(float).round(3).astype({'seeds': int}).astype(object)


def fmt(m, s=None, d=3):
    if not np.isfinite(m): return 'n/a'
    return f'{m:.{d}f}' + (f' ± {s:.{d}f}' if s is not None and np.isfinite(s) else '')


def fmt_ci(r, ci):
    return f'{r:.3f} [{ci[0]:.3f}; {ci[1]:.3f}]' if np.isfinite(r) else 'n/a'


def md(df):
    cols = [str(c) for c in df.columns]
    out = ['| ' + ' | '.join([df.index.name or ''] + cols) + ' |', '|---' * (len(cols)+1) + '|']
    for i, r in df.iterrows(): out.append('| ' + ' | '.join([str(i)] + [str(v) for v in r.values]) + ' |')
    return '\n'.join(out)


def all_tables():
    T = {}
    rows = []
    for name, fam in [('DV-FM', 'srv9_Ar02'), ('TF-2L', 'srv10_N2r02'), ('TF-6L', 'srv4_N6r02'), ('PD', 'srv5_Dcomb'),
                      ('PD-CVAE-S', 'srv7_Dzsh'), ('PD-CVAE-G', 'srv7_Dzgh')]:
        p = runs()[runs().family == fam].parameters.dropna()
        rows.append(dict(model=name, family=fam, parameters=f'{p.iloc[0]/1e3:.0f}k' if len(p) else 'n/a'))
    T['Models (parameters at test time)'] = pd.DataFrame(rows).set_index('model')

    rows = []
    for name, (r1, r02) in CAPACITY.items():
        a, b = family_skill(r1, 'test_ood'), family_skill(r02, 'test_ood')
        rows.append(dict(model=name, **{'rec. 1': fmt(a[0], a[1]), 'rec. 0.2': fmt(b[0], b[1]), 'seeds': f'{a[2]} / {b[2]}'}))
    T['Capacity: DV-FM vs its trend Transformer (TF-2L) and a same-size Transformer (TF-6L), OOD skill'] = pd.DataFrame(rows).set_index('model')

    rows = []
    for loss, (tf, pd_) in MAIN.items():
        for name, fam in (('TF-6L', tf), ('PD', pd_)):
            row = dict(loss=loss, model=name, seeds=family_skill(fam, 'test_ood')[2])
            for split, tag in (('test_ood', 'OOD'), ('test_id', 'ID')):
                m, s, _ = family_skill(fam, split); row[f'{tag} skill'] = fmt(m, s)
                row[f'{tag} ratio PD/TF-6L'] = fmt_ci(*ratio_ci(pd_, tf, split)) if name == 'PD' else ''
            rows.append(row)
    for name, fam in EXTRA.items():
        row = dict(loss='', model=name, seeds=family_skill(fam, 'test_ood')[2])
        for split, tag in (('test_ood', 'OOD'), ('test_id', 'ID')):
            m, s, _ = family_skill(fam, split); row[f'{tag} skill'] = fmt(m, s); row[f'{tag} ratio PD/TF-6L'] = ''
        rows.append(row)
    rows.append(dict(loss='', model='ES+Fourier (no training)', seeds='', **{'OOD skill': fmt(baseline_skill('es_fourier', 'test_ood')), 'OOD ratio PD/TF-6L': '',
                                                                           'ID skill': fmt(baseline_skill('es_fourier', 'test_id')), 'ID ratio PD/TF-6L': ''}))
    T['Main comparison: PD vs TF-6L at the same loss (skill mean ± sd over seeds; ratio with 95% CI)'] = pd.DataFrame(rows).set_index('loss')

    rows = []
    for label, num, den in [('PD-MixC / TF-6L-Mix + 50% KernelSynth', 'srv10_Dmc1', 'srv10_N6mix_ks50'),
                            ('PD / TF-6L + 50% KernelSynth', 'srv5_Dcomb', 'srv7_N6r02_ks50')]:
        rows.append(dict(comparison=label, **{SPLITS[s]: fmt_ci(*ratio_ci(num, den, s)) for s in SPLITS}))
    T['Against the plain model with synthetic data'] = pd.DataFrame(rows).set_index('comparison')

    rows = []
    for name, fams in KERNELSYNTH.items():
        for share, fam in zip((0, 25, 50, 100), fams):
            row = dict(model=name, kernelsynth_percent=share)
            for split, tag in (('test_ood', 'OOD'), ('test_id', 'ID')): m, s, n = family_skill(fam, split); row[tag] = fmt(m, s)
            rows.append(row)
    T['KernelSynth dose (skill mean ± sd over seeds)'] = pd.DataFrame(rows).set_index('model')

    rows = []
    for name, (tf, pd_) in SCALE.items():
        row = dict(setting=name, seeds=f'{family_skill(tf, "test_ood")[2]} / {family_skill(pd_, "test_ood")[2]}')
        for split, tag in (('test_ood', 'OOD'), ('test_id', 'ID')):
            row[f'{tag} TF-6L-MixC'] = fmt(*family_skill(tf, split)[:2]); row[f'{tag} PD-MixC'] = fmt(*family_skill(pd_, split)[:2])
            row[f'{tag} ratio'] = fmt_ci(*ratio_ci(pd_, tf, split))
        rows.append(row)
    T['Scale: PD-MixC vs TF-6L-MixC at width 64 / 1x data and width 128 / 16x data'] = pd.DataFrame(rows).set_index('setting')

    rows = []
    for name, fam in UNCERTAINTY.items():
        u = uncertainty_row(fam, 'test_ood')
        if u is None: continue
        rows.append(dict(model=name, skill=f'{u.skill:.3f}', PCE=f'{u.pce:.3f}', CCE=f'{u.cce:+.3f}', c50=f'{u.c50:.3f}', c80=f'{u.c80:.3f}',
                         c98=f'{u.c98:.3f}', ES=f'{u.nes:.2f}', VS=f'{u.nvs:.3f}', latent=f'{u.latent_share:.3f}'))
    T['Forecast distribution on OOD (mean over seeds; targets CCE 0, c50 .50, c80 .80, c98 .98)'] = pd.DataFrame(rows).set_index('model')

    lab = lab_table()
    T['Synthetic lab, models trained on the lab (wave share: P = 128 -> 1, P >= 192 -> 0; CRPS / true distribution)'] = lab

    rows = []
    for change, ref, num, den, col in ABLATIONS:
        o, i = ratio_by_seed(num, den, 'test_ood', col), ratio_by_seed(num, den, 'test_id', col)
        rows.append(dict(change=change, reference=ref, OOD=f'{o[0]:.3f} ({o[1]})' if o[1] else 'n/a', ID=f'{i[0]:.3f} ({i[1]})' if i[1] else 'n/a'))
    T['Ablations: skill ratio to the reference, median over seeds (number of seed pairs)'] = pd.DataFrame(rows).set_index('change')

    for split, title in SPLITS.items():
        fams = {'TF-6L': 'srv4_N6r02', 'PD': 'srv5_Dcomb', 'TF-6L-MixC': 'srv11_N6mc1', 'PD-MixC': 'srv10_Dmc1'}
        t = pd.DataFrame({k: per_source_skill(f, split) for k, f in fams.items()})
        t['PD-MixC / TF-6L-MixC'] = t['PD-MixC']/t['TF-6L-MixC']; t.index.name = 'source'
        T[f'Skill per source, {title} (seed means)'] = t.round(3)
    return T


def main():
    out = ['# Tables', '', 'Generated by `paper_tables.py` from the files in this folder.', '']
    for title, t in all_tables().items():
        out += [f'## {title}', '', md(t), '']
    text = '\n'.join(out) + '\n'; (RESULTS/'tables.md').write_text(text); print(text)


if __name__ == '__main__':
    main()
