import unittest

import torch
from torch import nn

from speed_limit.k3_gate import merge_adapter
from speed_limit.spectral import LoRALinear


class GateTests(unittest.TestCase):
    def test_merge_preserves_lora_forward_and_restores_base_keys(self):
        student = nn.Module()
        student.model = nn.Sequential(nn.Sequential(nn.Linear(7, 5)))
        keys = tuple(student.state_dict())
        layer = LoRALinear(student.model[0][0], rank=2, alpha=4)
        with torch.no_grad():
            layer.b.fill_(0.2)
        student.model[0][0] = layer
        tokens = torch.randn(4, 7)
        expected = student.model(tokens).detach()

        merge_adapter(student, ['0.0'])

        self.assertEqual(tuple(student.state_dict()), keys)
        torch.testing.assert_close(student.model(tokens), expected)
