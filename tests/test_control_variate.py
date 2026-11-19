import unittest
from types import SimpleNamespace

import torch

from speed_limit.control_variate import rollout_loss


class ControlVariateTest(unittest.TestCase):
    def test_fixed_rollout_sign_alignment_and_detachment(self):
        generator = torch.Generator().manual_seed(117)

        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(32, 32, generator=generator))

            def forward(self, input_ids, use_cache):
                return SimpleNamespace(logits=self.weight[input_ids])

        student, teacher = Toy(), Toy()
        contexts = [4, 7, 6]
        labels = torch.tensor([7, 6, 3])
        target = teacher.weight.detach()[contexts].log_softmax(-1)
        sampled = target.gather(-1, labels[:, None]).flatten()
        record = {'prompt': [2, 4], 'response': labels.tolist(), 'teacher_logprobs': sampled}
        for top_k in (None, 20):
            with self.subTest(top_k=top_k):
                loss, baseline = rollout_loss(student, teacher, record, top_k)
                logits = student.weight.detach()[contexts]
                student_logp = logits.log_softmax(-1)
                if top_k:
                    ids = logits.topk(top_k, dim=-1).indices
                    p = student_logp.gather(-1, ids).log_softmax(-1)
                    q = target.gather(-1, ids).log_softmax(-1)
                else:
                    p, q = student_logp, target
                kl = torch.distributions.kl_divergence(
                    torch.distributions.Categorical(logits=p),
                    torch.distributions.Categorical(logits=q))
                torch.testing.assert_close(baseline, kl.sum())
                advantage = sampled - student_logp.gather(-1, labels[:, None]).flatten() + kl
                score = torch.nn.functional.one_hot(labels, 32) - student_logp.exp()
                expected = torch.zeros_like(student.weight)
                expected[contexts] = -advantage[:, None] * score
                gradient, teacher_gradient = torch.autograd.grad(
                    loss, (student.weight, teacher.weight), allow_unused=True)
                torch.testing.assert_close(gradient, expected, atol=2e-6, rtol=1e-5)
                self.assertIsNone(teacher_gradient)
