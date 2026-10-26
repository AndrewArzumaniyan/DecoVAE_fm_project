"""Mini-FM training on the leave-sources-out ver0 corpus, reusing decomp_benchmark.runner.

Arms: A (the original MA + Top-K), Cf (period-aware input + continuous Fourier decoder), and
Cf0 (Cf trained and evaluated with the learned correction Δ switched off; same initialisation as Cf), Cs1/Cs2/Cs12 (Cf0 plus learnable horizon-only seasonal corrections, see seasonal_arms.py), N (no decomposition: the original patch-Transformer VAE on raw x, mean and log-variance heads,
4 encoder layers) and N6 (the same with 6 layers, 318k parameters, capacity-matched to A and Cf). Loss, optimiser, normalisation, checkpoint selection and resume are runner's.
After training, the best checkpoint is also evaluated zero-shot on held-out sources (test_ood).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

from fm_paths import ROOT, VER0_DIR, RUNS, CACHE  # noqa: E402,F401  (also puts the shared benchmark modules on sys.path)
import runner
from data import T, L, normalize
from adapters import Arm, _build_mlp
from fourier_arm import FOURIER_ARMS, attach, make_model as base_make_model
from decovaefm.models.decovae_fm import DecoVAEFM, TrendEncoder
import seasonal_arms as sa
import trend_input_arms as ta
import cvae_arms as ca
import score_arms as sc
import hp_arms as hp

OUT = RUNS
SPLITS = ('train', 'val', 'test_id', 'test_ood')


class NoDecomposition(nn.Module):
    """Same patch Transformer and heads as the trend stream, applied to the raw window."""
    def __init__(self, d_model=64, patch=16, layers=4):
        super().__init__()
        self.arm = 'N'; self.patch = patch
        self.encoder = TrendEncoder(patch, patch, d_model, 4, layers)
        self.mu_decoder = _build_mlp(d_model, d_model, patch, 2)
        self.logvar_decoder = _build_mlp(d_model, d_model, patch, 2)
        nn.init.zeros_(self.logvar_decoder[-1].bias)
        with torch.no_grad(): self.logvar_decoder[-1].weight.mul_(.01)

    def forward(self, x):
        n = x.shape[-1]; tm, tl = self.encoder(x, n_target=L//self.patch)
        z = tm + torch.randn_like(tm)*torch.exp(.5*tl.clamp(-20, 10))  # sampled at eval too, as in DecoVAEFM
        mu = self.mu_decoder(z).flatten(1); lv = self.logvar_decoder(z).flatten(1).clamp(-20, 10)
        return dict(trend_mu=tm, trend_logvar=tl, recon_mu=mu[:, :n], recon_logvar=lv[:, :n],
                    forecast_mu=mu[:, n:], forecast_logvar=lv[:, n:], decoded_trend=mu, decoded_seasonal=torch.zeros_like(mu))


class NoDecompositionMix(NoDecomposition):
    """N6mix: N6 trained like Dmix, with the log score of the K = 8 latent-draw mixture as the forecast loss
    (the control for Dmix: same loss without decomposition). Encoder once, decoders K times; evaluation is N6's forward."""
    kind, lam = 'mix', 0.

    def forward(self, x):
        if not self.training: return super().forward(x)
        n = x.shape[-1]; tm, tl = self.encoder(x, n_target=L//self.patch); mus, lvs = [], []
        for _ in range(sc.K_DRAWS):
            z = tm+torch.randn_like(tm)*torch.exp(.5*tl.clamp(-20, 10))
            mus.append(self.mu_decoder(z).flatten(1)); lvs.append(self.logvar_decoder(z).flatten(1).clamp(-20, 10))
        mu, lv = torch.stack(mus), torch.stack(lvs)
        return dict(trend_mu=tm, trend_logvar=tl, recon_mu=mu[0, :, :n], recon_logvar=lv[0, :, :n], forecast_mu=mu[0, :, n:], forecast_logvar=lv[0, :, n:],
                    decoded_trend=mu[0], decoded_seasonal=torch.zeros_like(mu[0]), mix_mu=mu[:, :, n:], mix_lv=lv[:, :, n:], score_kind=self.kind, score_lam=self.lam)


# Optional model-size overrides (--d-model, --enc-layers, --k-top, --n-layers); empty = original sizes, bit for bit.
SIZE = {}
_decovae_init = DecoVAEFM.__init__


def _sized_init(self, *args, **kwargs):
    kwargs.update({k: v for k, v in SIZE.items() if k in ('d_model', 'enc_layers', 'k_top')}); _decovae_init(self, *args, **kwargs)


DecoVAEFM.__init__ = _sized_init


def make_model(arm):
    if arm in ('N', 'N6'):
        return NoDecomposition(d_model=SIZE.get('d_model', 64), layers=SIZE.get('n_layers', 6 if arm == 'N6' else 4))
    if arm in ('N6mix', 'N6mc1'):
        m = NoDecompositionMix(d_model=SIZE.get('d_model', 64), layers=SIZE.get('n_layers', 6))
        if arm == 'N6mc1': m.kind, m.lam = 'mixcrps', 1.  # N6 with Dmc1's loss (control for Dmc1)
        return m
    if arm == 'Cf0':
        model = base_make_model('Cf'); model.use_correction = False; return model
    if arm in sa.S_ARMS: return sa.AdaptiveSeasonalArm(arm)
    if arm in ta.D_ARMS: return ta.DeseasonalizedArm(arm)
    if arm in sa.SIG_ARMS: return sa.SigmaInfoArm(arm)
    if arm in ta.COMB_ARMS: return ta.CombinedArm(arm)
    if arm in ca.Z_ARMS: return ca.CVAEArm(arm)
    if arm in sc.SCORE_ARMS: return sc.ScoreArm(arm)
    if arm in hp.HP_ARMS: return hp.HPArm(arm)
    if arm in hp.CAP_ARMS: return hp.CappedArm(arm)
    return base_make_model(arm)


_set_batch = runner.set_batch


def _set_batch_s12(model, raw, idx, device):
    sa.set_s12(model, raw, idx, device)
    if hasattr(model, 'future'): model.future = torch.from_numpy(raw['z'][idx, T:]).to(device)  # CVAE posterior input (training only)
    return _set_batch(model, raw, idx, device)


runner.set_batch = _set_batch_s12  # adds the S1/S2 inputs for the adaptive seasonal arms; no-op for other arms
runner.loss = ca.wrap_loss(runner.loss)  # adds KL(q || p) / L for the CVAE arms; unchanged for the others
runner.loss = sc.wrap_loss(runner.loss)  # forecast term = score of the latent mixture for the score arms; unchanged for the others
runner.loss = hp.wrap_loss(runner.loss)  # HP trend penalty for the HP arms; unchanged for the others


def load_split(cache, split, arm, clip):
    raw = dict(np.load(cache/f'{split}.npz'))
    z, m, s = normalize(raw['x'].astype(np.float64))
    if clip: z = np.clip(z, -clip, clip)
    raw.update(z=z.astype('float32'), mean=m, scale=s)
    if arm in FOURIER_ARMS or arm == 'Cf0' or arm in sa.S_ARMS or arm in ta.D_ARMS or arm in sa.SIG_ARMS or arm in ta.COMB_ARMS or arm in ca.Z_ARMS or arm in sc.SCORE_ARMS or arm in hp.HP_ARMS or arm in hp.CAP_ARMS: attach(raw, cache, split, stop=lambda: runner.STOP_REQUESTED)
    if arm in sa.S_ARMS or arm in sa.SIG_ARMS or arm in ta.COMB_ARMS or arm in ca.Z_ARMS or arm in sc.SCORE_ARMS or arm in hp.HP_ARMS or arm in hp.CAP_ARMS: sa.attach_s12(raw, cache, split)
    return raw


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True); p.add_argument('--arm', choices=['A', 'N', 'N6', 'N6mix', 'N6mc1', 'Cf0', *FOURIER_ARMS, *sa.S_ARMS, *ta.D_ARMS, *sa.SIG_ARMS, *ta.COMB_ARMS, *ca.Z_ARMS, *sc.SCORE_ARMS, *hp.HP_ARMS, *hp.CAP_ARMS], required=True)
    p.add_argument('--seed', type=int, default=0); p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--minutes', type=float, default=600); p.add_argument('--device', default='cuda')
    p.add_argument('--clip', type=float, default=0.); p.add_argument('--run-id', required=True)
    p.add_argument('--resume'); p.add_argument('--batch', type=int, default=32); p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--patience', type=int, default=10, help='in validations'); p.add_argument('--val-every', type=int, default=1, help='epochs between validations')
    p.add_argument('--d-model', type=int); p.add_argument('--enc-layers', type=int); p.add_argument('--k-top', type=int)
    p.add_argument('--n-layers', type=int, help='layers of the no-decomposition arms N/N6')
    p.add_argument('--recon-weight', type=float, default=1.0, help='weight of the context-reconstruction NLL')
    p.add_argument('--fast-val', action='store_true', help='vectorised validation sampling (opt-in, changes only the validation draws)')
    p.add_argument('--val-limit', type=int); p.add_argument('--test-limit', type=int)
    a = p.parse_args()
    cache = Path(a.cache)
    runner.RECON_WEIGHT = a.recon_weight
    SIZE.update({k: v for k, v in dict(d_model=a.d_model, enc_layers=a.enc_layers, k_top=a.k_top, n_layers=a.n_layers).items() if v is not None})
    args = argparse.Namespace(command='train', profile='fm_mini', arm=a.arm, seed=a.seed, device=a.device, real=False, n=0,
                              epochs=a.epochs, minutes=a.minutes, val_limit=a.val_limit, test_limit=a.test_limit,
                              resume=a.resume, run_id=a.run_id, batch=a.batch, lr=a.lr, patience=a.patience, val_every=a.val_every, fast_val=a.fast_val, recon_weight=a.recon_weight, size=dict(SIZE), data='real', cache=str(cache), clip=a.clip)
    path = OUT/a.run_id
    runner.OUT = OUT
    runner.make_model = make_model

    def load_splits(_):
        return {'train': load_split(cache, 'train', a.arm, a.clip), 'val': load_split(cache, 'val', a.arm, a.clip),
                'test': load_split(cache, 'test_id', a.arm, a.clip)}, cache
    runner.load_splits = load_splits
    runner.provenance(path, args)
    runner.train(args, path)
    status = json.loads((path/'status.json').read_text())
    if status.get('status') != 'complete': return
    # Zero-shot evaluation on held-out sources with the selected checkpoint.
    model = make_model(a.arm).to(a.device)
    model.load_state_dict(torch.load(path/'best.pt', weights_only=False, map_location=a.device)['model'])
    ood = load_split(cache, 'test_ood', a.arm, a.clip)
    df, outputs = runner.evaluate(model, ood, a.device, 128, a.test_limit)
    df['dataset'] = ood['dataset'][:len(df)]; df['arm'] = a.arm; df['seed'] = a.seed
    df['ncrps'] = df.crps/ood['scale'][:len(df), 0]; df['nmae'] = df.mae/ood['scale'][:len(df), 0]
    df.to_csv(path/'ood_rows.csv', index=False)
    np.savez_compressed(path/'ood_outputs.npz', mean=np.concatenate([o['mean'] for o in outputs]),
                        q05=np.concatenate([np.quantile(o['samples'], .05, axis=0) for o in outputs]),
                        q95=np.concatenate([np.quantile(o['samples'], .95, axis=0) for o in outputs]),
                        decoded=np.concatenate([o['decoded_normalized'] for o in outputs]),
                        # PIT per point from the 128 predictive samples (calibration shape: any coverage level, tails, PCE).
                        pit=np.concatenate([(o['samples'] < ood['x'][o['indices'], T:][None]).mean(0) for o in outputs]).astype('float32'))
    params = sum(v.numel() for v in model.parameters())
    status.update(ood_evaluated=True, parameters=params); (path/'status.json').write_text(json.dumps(status, indent=2))
    print('ood', df.groupby('dataset').ncrps.mean().to_dict(), flush=True)


if __name__ == '__main__':
    main()
