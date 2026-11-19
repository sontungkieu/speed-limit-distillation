import unittest

import torch
from torch import nn

from speed_limit.k3 import curve_agreement
from speed_limit.spectral import LoRALinear, StreamedGram, residual_prediction


class SpectralTests(unittest.TestCase):
    def test_curve_agreement_keeps_later_steps_out_of_calibration(self):
        reference = [1.] * 101
        calibration_fit = [1.] * 51 + [.5] * 50
        agreement = curve_agreement(calibration_fit, reference)
        self.assertEqual(agreement['calibration_rmse_1_50'], 0.)
        self.assertEqual(agreement['validation_rmse_51_100'], .5)
        self.assertEqual(agreement['difference_at_100'], -.5)
        with self.assertRaises(ValueError):
            curve_agreement(calibration_fit[:-1], reference[:-1])

    def test_streamed_gram_matches_explicit_token_gradients(self):
        class StateLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([.4, -.5]))

            def forward(self, batch):
                return (batch[2] @ self.weight).square()

        model = StateLoss()
        states = [torch.tensor([[1., 2.], [3., 1.]]), torch.tensor([[-1., 5.]])]
        batches = [(None, None, state, None) for state in states]
        scale = torch.tensor([2., 4.])
        gram = StreamedGram(model, batches, {'weight': scale})
        x = torch.cat(states)
        jacobian = 2 * (x @ model.weight.detach())[:, None] * x
        u = torch.tensor([.2, -.1, .7])
        expected = jacobian @ (scale * (jacobian.T @ u)) / len(states)
        torch.testing.assert_close(gram.matvec(u), expected)

    def test_adapter_preserves_step_zero_and_proxy_tracks_unresolved_mass(self):
        torch.manual_seed(10)
        layer = nn.Linear(3, 4).requires_grad_(False)
        adapter = LoRALinear(layer, rank=2, alpha=4)
        x = torch.randn(3, 3)
        torch.testing.assert_close(adapter(x), layer(x), atol=0, rtol=0)
        adapter(x).sum().backward()
        self.assertIsNone(layer.weight.grad)
        self.assertGreater(adapter.b.grad.norm(), 0)
        _, unresolved, curve = residual_prediction(torch.tensor([1., 1.]), torch.tensor([1.]),
                                                   torch.tensor([[1.], [0.]]), .5, 2)
        torch.testing.assert_close(unresolved, torch.tensor(.5, dtype=torch.float64))
        torch.testing.assert_close(curve, torch.tensor([1., .625, .53125], dtype=torch.float64))


if __name__ == '__main__':
    unittest.main()
