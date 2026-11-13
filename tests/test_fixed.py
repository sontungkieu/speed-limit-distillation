import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from speed_limit.fixed import capture_training_state, restore_training_state, read_curve


class FixedStateTests(unittest.TestCase):
    def test_checkpoint_resume_matches_continuous_stochastic_updates(self):
        random.seed(23)
        np.random.seed(23)
        torch.manual_seed(23)
        model = torch.nn.Linear(3, 1, dtype=torch.float64)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)

        def update(network, algorithm):
            inputs = torch.randn(5, 3, dtype=torch.float64)
            target = torch.tensor(np.random.randn(5, 1), dtype=torch.float64) + random.random()
            algorithm.zero_grad(set_to_none=True)
            loss = (network(inputs) - target).square().mean()
            loss.backward()
            algorithm.step()

        for _ in range(4):
            update(model, optimizer)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.pt'
            torch.save({'model': model.state_dict(),
                        'training': capture_training_state(optimizer, 4)}, path)
            for _ in range(4):
                update(model, optimizer)
            expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
            restored = torch.nn.Linear(3, 1, dtype=torch.float64)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.001)
            saved = torch.load(path, map_location='cpu', weights_only=True)
            restored.load_state_dict(saved['model'])
            self.assertEqual(restore_training_state(restored_optimizer, saved['training']), 4)
            for _ in range(4):
                update(restored, restored_optimizer)
            for name, value in restored.state_dict().items():
                self.assertTrue(torch.equal(value, expected[name]), name)

    def test_resume_history_keeps_last_measurement_per_step(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'curve.jsonl'
            records = [{'step': 0, 'A': 1.}, {'step': 50, 'A': .2},
                       {'step': 50, 'A': .19}, {'step': 51, 'A': .18}]
            path.write_text(''.join(json.dumps(row) + '\n' for row in records))
            self.assertEqual(read_curve(path, 50), [records[0], records[2]])


if __name__ == '__main__':
    unittest.main()
