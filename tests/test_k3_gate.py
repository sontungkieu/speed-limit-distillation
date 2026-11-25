import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
from torch import nn

from speed_limit.k3_gate import merge_adapter, reuse_dev
from speed_limit.legacy import Experiment
from speed_limit.spectral import LoRALinear


class GateTests(unittest.TestCase):
    def test_token_cap_only_applies_to_dev(self):
        experiment = Experiment()
        experiment.refit = Mock()
        experiment.render = Mock(return_value=[1337])
        experiment.generate = Mock(side_effect=InterruptedError)
        experiment.benchmarks = {'aime24': [{'problem': 'test'}]}
        for dev, canary, cap, expected in ((True, False, 2048, 2048), (True, False, None, 16384),
                                             (False, False, 2048, 32768), (True, True, 2048, 64)):
            experiment.args = SimpleNamespace(dev=dev, canary=canary, dev_max_tokens=cap, seed=0)
            with self.assertRaises(InterruptedError):
                experiment.evaluate(0)
            self.assertEqual(experiment.generate.call_args.args[1], expected)

    def test_recovers_partial_dev_and_validates_finished_journal(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prior = root / 'prior'
            (prior / 'dev').mkdir(parents=True)
            args = SimpleNamespace(prior=prior, inputs=root / 'inputs', output=root / 'new-dev')
            contract = {'checkpoint': str(prior / 'merged'), 'checkpoint_step': 300, 'task': 'evaluate',
                        'seed': 0, 'dev': True, 'canary': False, 'inputs': str(args.inputs)}
            (prior / 'dev' / 'started.json').write_text(json.dumps(contract))
            journal = prior / 'dev' / 'eval-step-0300.jsonl'
            journal.touch()
            self.assertFalse(reuse_dev(args, 300))
            self.assertEqual(args.resume_eval, journal)

            names = ('aime24', 'aime25')
            rows = ({'benchmark': name, 'problem_index': i, 'sample': sample, 'seed': 0,
                     'checkpoint_step': 300, 'correct': True}
                    for name in names for i in range(30) for sample in range(8))
            journal.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            summary = {'samples_per_problem': 8, 'benchmarks': {
                name: {'correct': 240, 'total': 240} for name in names}}
            (prior / 'dev' / 'eval-step-0300.json').write_text(json.dumps(summary))
            self.assertTrue(reuse_dev(args, 300))
            self.assertEqual((args.output / journal.name).read_bytes(), journal.read_bytes())

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
