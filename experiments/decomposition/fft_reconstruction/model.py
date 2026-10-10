"""Change only the final Fourier-to-time operation of the repository model.

No period estimator, deterministic seasonal baseline, extra parameters or loss terms.
The decoded complex coefficients retain the length-T rFFT convention in all arms.
"""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from decovaefm.models.decovae_fm import DecoVAEFM

VARIANTS = ("original", "amplitude", "exact_frequency")


def exact_frequency_reconstruction(tokens, indices, context_len, output_length):
    """Evaluate the same rFFT coefficients at k/T, normalized by T.

    This is the T-periodic extension of the sparse inverse FFT, *not* estimation
    of the signal's true off-grid period. DC and the even-T Nyquist coefficient
    have multiplicity one and no imaginary contribution; other bins have two.
    """
    coefficients = tokens.squeeze(-1)
    real_dtype = coefficients.real.dtype
    time = torch.arange(output_length, device=coefficients.device, dtype=torch.float64)
    cycles = indices.to(torch.float64)[..., None] * time / context_len
    phase = (2 * torch.pi * torch.remainder(cycles, 1)).to(real_dtype)
    endpoint = indices == 0
    if context_len % 2 == 0:
        endpoint = endpoint | (indices == context_len // 2)
    multiplicity = torch.where(endpoint, 1., 2.).to(real_dtype)
    imaginary = coefficients.imag.masked_fill(endpoint, 0)
    waves = (coefficients.real[..., None] * phase.cos()
             - imaginary[..., None] * phase.sin())
    return (waves * multiplicity[..., None]).sum(dim=1) / context_len


class FFTReconstructionArm(DecoVAEFM):
    """Native forward/encode/decoders; override just reconstruction."""

    def __init__(self, variant="original", pred_len=96, **kwargs):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown FFT reconstruction variant: {variant}")
        super().__init__(pred_len=pred_len, **kwargs)
        self.variant = variant

    def _reconstruct_time_from_freq_tokens(self, tokens, indices, context_len, output_length):
        if self.variant == "exact_frequency":
            return exact_frequency_reconstruction(tokens, indices, context_len, output_length)
        if self.variant == "amplitude":
            tokens = tokens * (output_length / context_len)
        return super()._reconstruct_time_from_freq_tokens(tokens, indices, context_len, output_length)
