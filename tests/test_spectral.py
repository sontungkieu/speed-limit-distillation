import unittest

import torch
from torch import nn

from speed_limit.k3 import checkpoint_preconditioner, curve_agreement
from speed_limit.lanczos import lanczos_top_eigenpairs
from speed_limit.residual_k3 import EPS, project_sgd
from speed_limit.spectral import LoRALinear, StreamedGram, residual_prediction


class SpectralTests(unittest.TestCase):
    def test_residual_start_recovers_the_residual_weighted_dense_sgd_curve(self):
        gram = torch.diag(torch.tensor([2., 8.], dtype=torch.float64))
        residual = torch.tensor([3., 4.], dtype=torch.float64)
        projection = lanczos_top_eigenpairs(lambda u: gram @ u, 2, iterations=2,
            dtype=torch.float64, initial_vector=residual, return_projection=True)
        result = project_sgd(projection, 5., 2, 2, .02, 4)
        state = residual.clone()
        velocity = torch.zeros_like(state)
        expected = [1.]
        for _ in range(4):
            clip = min(1., 1. / (float((state @ gram @ state / 2).sqrt()) + 1e-6))
            velocity = .9 * velocity + clip * gram @ state + .01 * (state - residual)
            state -= .02 * velocity
            expected.append(float((state.square().sum() - 2 * EPS) / (25 - 2 * EPS)))
        torch.testing.assert_close(torch.tensor(result['curve']), torch.tensor(expected))
        self.assertAlmostEqual(sum(result['residual_weights']), 1.)

    def test_saved_adam_moments_follow_adapter_parameter_order(self):
        parameters = {'a': torch.zeros(2), 'b': torch.zeros(1)}
        group = {'params': [7, 3], 'lr': 3e-5, 'betas': (.9, .5), 'eps': 0.}
        state = {3: {'step': torch.tensor(2.), 'exp_avg_sq': torch.tensor([12.])},
                 7: {'step': torch.tensor(2.), 'exp_avg_sq': torch.tensor([.75, 3.])}}
        optimizer = {'param_groups': [group], 'state': state}
        p = checkpoint_preconditioner(parameters, optimizer, 2, 3e-5)
        torch.testing.assert_close(p['a'], torch.tensor([1., .5]))
        torch.testing.assert_close(p['b'], torch.tensor([.25]))
        with self.assertRaises(ValueError):
            checkpoint_preconditioner(parameters, optimizer, 1, 3e-5)

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
