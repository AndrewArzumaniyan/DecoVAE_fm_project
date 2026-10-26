"""Train Dcomb's forecast with a proper score of the latent mixture instead of E_z[NLL].

The predictive of the VAE is the mixture over latent draws z ~ q(z | context) of N(mu(z), sd(z)^2). The usual loss, the
NLL of one draw, penalises any spread of mu(z) across draws (Jensen), so the latents collapse. Here the encoder runs once,
the decoders K = 8 times (independent z draws, shared dropout of the encoder), and the forecast term is
- Dcrps: almost fair CRPS of the K-component Gaussian mixture, closed form (Grimit et al. 2006) with the pair term
  averaged over i != j (fair) mixed with the plug-in term over all pairs, alpha = 0.95 as in AIFS-CRPS (Lang et al. 2024):
  mean_i A(y - mu_i, s_i^2) - 1/2 [alpha sum_{i!=j} A_ij / (K (K - 1)) + (1 - alpha) sum_{i,j} A_ij / K^2],
  A(m, v) = m (2 Phi(m / sqrt v) - 1) + 2 sqrt v phi(m / sqrt v), A_ij = A(mu_i - mu_j, s_i^2 + s_j^2);
- Dmix: log score of the same K mixture per point, -log (1/K sum_i N(y; mu_i, s_i^2)) (constants as in runner.loss).
- Dmc1 / Dmc3 / Dmc10 (16.10): Dmix + lambda * Dcrps, lambda = 1 / 3 / 10 (a sum of two proper scores is proper for any
  lambda); Dmix was the better calibrated (OOD tails), Dcrps the better on joint paths and ID calibration.
- DmixC (16.10): Dmix with the log-variances capped at 2, for the asinh-space cache (see hp_arms.CappedArm).
- DmixK4 / DmixK16 (17.10): Dmix with K = 4 / 16 latent draws per step (sensitivity to K).
- Dmixj (17.10): the joint (path) log score, -log (1/K sum_k prod_t N(y_t; mu_k,t, s_k,t^2)) / L: the K-sample importance
  bound (Burda et al. 2016) of the whole 96-step path with q(z | context) as the proposal; it rewards draws that are
  coherent paths, while the per-point version only rewards the marginals.
- Dmc1s (17.10): Dmc1 plus the learnable horizon-only seasonal corrections S1 (recency-weighted coefficients) and S2
  (trust gate) of seasonal_arms.AdaptiveSeasonalArm, zero-initialised.
Both reward spread of the draws when it improves the score; with collapsed latents they reduce to the Gaussian CRPS /
NLL of one component. The context reconstruction keeps the NLL of the first draw (weight 0.2) and the 0.01 KL, as Dcomb.
At evaluation the model is Dcomb's forward (one draw per call), so the test predictive is the same latent mixture.
"""
import math
import torch
from data import L
from trend_input_arms import CombinedArm

SCORE_ARMS = ('Dcrps', 'Dmix', 'Dmc1', 'Dmc3', 'Dmc10', 'DmixC', 'DmixK4', 'DmixK16', 'Dmixj', 'Dmc1s')
LAMBDA = {'Dmc1': 1., 'Dmc3': 3., 'Dmc10': 10., 'Dmc1s': 1.}
K_ARM = {'DmixK4': 4, 'DmixK16': 16}
LV_MAX = 2.
K_DRAWS = 8
ALPHA = .95


class ScoreArm(CombinedArm):
    def __init__(self, arm):
        super().__init__('Dcomb'); self.name = arm
        self.kind = 'crps' if arm == 'Dcrps' else 'mixcrps' if arm in LAMBDA else 'mixj' if arm == 'Dmixj' else 'mix'
        self.lam = LAMBDA.get(arm, 0.); self.cap = arm == 'DmixC'; self.k = K_ARM.get(arm, K_DRAWS); self.s12_corr = arm == 'Dmc1s'
        if self.s12_corr:
            from seasonal_arms import N_FEAT, HLS
            self.adapt = torch.nn.Sequential(torch.nn.Linear(N_FEAT, 32), torch.nn.GELU(), torch.nn.Linear(32, len(HLS)+1))
            torch.nn.init.zeros_(self.adapt[-1].weight); torch.nn.init.zeros_(self.adapt[-1].bias)

    def seasonal_path(self, zs, length):
        if not self.s12_corr: return super().seasonal_path(zs, length)
        from fourier_arm import fourier_basis, harmonics, N_SEASONS
        periods, coefs = self.fourier; coefs_hl, feats = self.s12
        pooled = torch.cat([zs.real, zs.imag], -1).mean(1)
        delta = self.fourier_head(torch.cat([pooled[:, None, :].expand(-1, N_SEASONS, -1), torch.log1p(periods.float())[..., None]], -1))
        _, mask = harmonics(periods.double()); c = coefs.float()+delta*mask.repeat_interleave(2, -1).float()
        basis = fourier_basis(periods, length); ctx = length-L; out = self.adapt(feats); nh = out.shape[1]-1
        c_h = c+torch.einsum('bk,bskh->bsh', torch.tanh(out[:, :nh]), coefs_hl.float()-c[:, :, None, :])   # S1
        s_h = (1+torch.tanh(out[:, -1:]))*torch.einsum('bsnk,bsk->bn', basis[:, :, ctx:, :], c_h)            # S2
        return torch.cat([torch.einsum('bsnk,bsk->bn', basis[:, :, :ctx, :], c), s_h], 1)

    def _decode(self, e, length):
        """One latent draw through Dcomb's decoders (same operations as DeseasonalizedArm/CombinedArm.forward)."""
        zt = self.reparameterize(e['trend_mu'], e['trend_logvar']); zs = self.reparameterize(e['seasonal_mu'], e['seasonal_logvar'])
        trend = self._fold_trend_patches(self.trend_decoder(zt), length+L)
        if self.anchor: trend = trend+self.level_anchor(e['target_trend'])[:, None]
        seasonal = self.seasonal_path(zs, length+L)
        pooled = torch.cat([zs.real, zs.imag], -1).mean(1, keepdim=True).expand(-1, zt.size(1), -1)
        lv = self._fold_trend_patches(self.logvar_decoder(torch.cat([pooled, zt], -1)), length+L).clamp(-20, 10)
        feats = self.s12[1]
        h = self.sig(torch.cat([seasonal[..., None], seasonal.abs()[..., None], feats[:, None, :].expand(-1, seasonal.shape[1], -1)], -1)).squeeze(-1)
        return trend+seasonal, (lv+h).clamp(-20, LV_MAX if self.cap else 10), trend, seasonal

    def forward(self, x, pred_len=None, noise_permutation=None):
        if not self.training:
            out = super().forward(x, pred_len, noise_permutation)
            if self.cap: out['recon_logvar'] = out['recon_logvar'].clamp(max=LV_MAX); out['forecast_logvar'] = out['forecast_logvar'].clamp(max=LV_MAX)
            return out
        n = x.shape[-1]; e = self.encode(x, n_target=L//16)
        draws = [self._decode(e, n) for _ in range(self.k)]
        mu = torch.stack([d[0] for d in draws]); lv = torch.stack([d[1] for d in draws])
        return dict(e, recon_mu=mu[0, :, :n], recon_logvar=lv[0, :, :n], forecast_mu=mu[0, :, n:], forecast_logvar=lv[0, :, n:],
                    decoded_trend=draws[0][2], decoded_seasonal=draws[0][3], mix_mu=mu[:, :, n:], mix_lv=lv[:, :, n:], score_kind=self.kind, score_lam=self.lam)


def _A(m, var):
    s = var.sqrt(); u = m/s
    return m*(2*torch.special.ndtr(u)-1)+2*s*torch.exp(-.5*u**2)/math.sqrt(2*math.pi)


def af_crps(y, mu, lv, alpha=ALPHA):
    """Almost fair CRPS of the equal-weight Gaussian mixture; mu, lv (K, B, L), y (B, L); mean over points."""
    var = lv.exp(); K = mu.shape[0]
    first = _A(y[None]-mu, var).mean(0)
    pairs = _A(mu[:, None]-mu[None], var[:, None]+var[None]).sum((0, 1))
    diag = (2*(2*var).sqrt()/math.sqrt(2*math.pi)).sum(0)            # A(0, 2 s_i^2), the i = j terms
    spread = alpha*(pairs-diag)/(K*(K-1))+(1-alpha)*pairs/K**2
    return (first-.5*spread).mean()


def mix_nll(y, mu, lv):
    ll = -.5*(lv+(y[None]-mu)**2*torch.exp(-lv))
    return -(torch.logsumexp(ll, 0)-math.log(mu.shape[0])).mean()


def mix_nll_joint(y, mu, lv):
    """Path log score of the K mixture, per point: -(logsumexp_k sum_t log N(y_t; mu_kt, s_kt^2) - log K) / L."""
    ll = (-.5*(lv+(y[None]-mu)**2*torch.exp(-lv))).sum(-1)
    return -(torch.logsumexp(ll, 0)-math.log(mu.shape[0])).mean()/mu.shape[-1]


def wrap_loss(base_loss):
    """Replace the forecast NLL of runner.loss by the score of the K-draw mixture for the score arms."""
    def loss(out, x, y):
        value = base_loss(out, x, y)
        if 'score_kind' in out:
            lv = out['forecast_logvar']; nll = .5*(lv+(y-out['forecast_mu'])**2*torch.exp(-lv)).mean()
            if out['score_kind'] == 'crps': score = af_crps(y, out['mix_mu'], out['mix_lv'])
            elif out['score_kind'] == 'mix': score = mix_nll(y, out['mix_mu'], out['mix_lv'])
            elif out['score_kind'] == 'mixj': score = mix_nll_joint(y, out['mix_mu'], out['mix_lv'])
            else: score = mix_nll(y, out['mix_mu'], out['mix_lv'])+out['score_lam']*af_crps(y, out['mix_mu'], out['mix_lv'])
            value = value-nll+score
        return value
    return loss
