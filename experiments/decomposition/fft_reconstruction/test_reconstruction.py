"""Numerical guarantees for the FFT-only ablation (stdlib unittest)."""
import unittest
import numpy as np
import torch

from model import FFTReconstructionArm, exact_frequency_reconstruction
from datasets import build_split, T
from decovaefm.models.decovae_fm import DecoVAEFM


class ReconstructionChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)

    def test_identity_at_context_length_even_and_odd(self):
        for length in (31, 32, 512, 513):
            x = torch.randn(2, length, dtype=torch.float64)
            coefficients = torch.fft.rfft(x)
            indices = torch.arange(coefficients.shape[1]).repeat(2, 1)
            result = exact_frequency_reconstruction(coefficients[..., None], indices, length, length)
            torch.testing.assert_close(result, x, atol=1e-12, rtol=1e-12)

    def test_dc_and_nyquist_endpoint_weights(self):
        indices = torch.tensor([[0, 16]])
        coefficients = torch.tensor([[[64 + 37j], [32 - 19j]]], dtype=torch.complex128)
        result = exact_frequency_reconstruction(coefficients, indices, 32, 57)
        expected = 2 + (-1.) ** torch.arange(57, dtype=torch.float64)
        torch.testing.assert_close(result[0], expected, atol=1e-12, rtol=1e-12)

    def test_original_is_native_model(self):
        torch.manual_seed(0)
        original = DecoVAEFM(pred_len=32, patch_size=16, stride=16, k_top=8).eval()
        arm = FFTReconstructionArm("original", pred_len=32, patch_size=16, stride=16, k_top=8).eval()
        arm.load_state_dict(original.state_dict())
        x = torch.randn(2, 128)
        torch.manual_seed(12)
        expected = original(x)
        torch.manual_seed(12)
        actual = arm(x)
        self.assertEqual(actual.keys(), expected.keys())
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)

    def test_only_seasonal_time_reconstruction_changes(self):
        original = FFTReconstructionArm("original", pred_len=32, patch_size=16, stride=16, k_top=8).eval()
        amplitude = FFTReconstructionArm("amplitude", pred_len=32, patch_size=16, stride=16, k_top=8).eval()
        exact = FFTReconstructionArm("exact_frequency", pred_len=32, patch_size=16, stride=16, k_top=8).eval()
        x = torch.randn(2, 128)
        torch.manual_seed(12)
        baseline = original(x)
        for arm in (amplitude, exact):
            arm.load_state_dict(original.state_dict())
            torch.manual_seed(12)
            out = arm(x)
            for key in baseline.keys() - {"recon_mu", "forecast_mu", "decoded_seasonal"}:
                torch.testing.assert_close(out[key], baseline[key], atol=0, rtol=0)
        torch.manual_seed(12)
        corrected = amplitude(x)
        torch.testing.assert_close(corrected["decoded_seasonal"], baseline["decoded_seasonal"] * 160 / 128)

    def test_amplitude_only_on_aligned_bins(self):
        t = torch.arange(608, dtype=torch.float64)
        truth = torch.cos(2 * torch.pi * t / 32)
        coeff = torch.fft.rfft(truth[:512])[16].reshape(1, 1, 1)
        indices = torch.tensor([[16]])
        for variant, scale in (("original", 512/608), ("amplitude", 1), ("exact_frequency", 1)):
            arm = FFTReconstructionArm(variant)
            out = arm._reconstruct_time_from_freq_tokens(coeff, indices, 512, 608)
            torch.testing.assert_close(out[0], truth * scale, atol=1e-12, rtol=1e-12)

    def test_frequency_rounding_depends_on_horizon(self):
        indices = torch.tensor([[8]])
        coefficients = torch.tensor([[[256 + 0j]]], dtype=torch.complex128)
        for horizon in (96, 128, 256):
            length = 512 + horizon
            truth = torch.cos(2 * torch.pi * torch.arange(length, dtype=torch.float64) / 64)
            exact = exact_frequency_reconstruction(coefficients, indices, 512, length)
            torch.testing.assert_close(exact[0], truth, atol=1e-12, rtol=1e-12)
            amp = FFTReconstructionArm("amplitude")._reconstruct_time_from_freq_tokens(
                coefficients, indices, 512, length)[0]
            if horizon == 96:
                self.assertGreater(float((amp - truth).square().mean().sqrt()), .5)
            else:
                torch.testing.assert_close(amp, truth, atol=1e-12, rtol=1e-12)

    def test_complex_gradients(self):
        coefficients = torch.randn(2, 3, 1, dtype=torch.complex128, requires_grad=True)
        indices = torch.tensor([[0, 3, 16], [1, 5, 16]])
        self.assertTrue(torch.autograd.gradcheck(
            lambda z: exact_frequency_reconstruction(z, indices, 32, 57), (coefficients,)))

    def test_horizons_share_paths_and_context_only_normalization(self):
        raw = build_split(8, 719, 256)
        shorter = build_split(8, 719, 96)
        np.testing.assert_array_equal(raw["x"][:, :T + 96], shorter["x"])
        np.testing.assert_array_equal(raw["z"][:, :T + 96], shorter["z"])
        np.testing.assert_allclose(raw["mean"], raw["x"][:, :T].mean(1, keepdims=True))
        np.testing.assert_allclose(raw["scale"], raw["x"][:, :T].std(1, keepdims=True))
        for key in ("x", "z", "trend", "seasonal", "noise"):
            self.assertEqual(raw[key].shape, (8, T + 256))
        np.testing.assert_allclose(raw["x"], raw["trend"] + raw["seasonal"] + raw["noise"], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
