import runpy
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch


preflight = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/run_residual_k3.py'))['gpu_preflight']


class ResidualLauncherTests(unittest.TestCase):
    def setUp(self):
        self.options = SimpleNamespace(gpu='GPU-free', mps_server=222, mps_pipe=Path('/mps/pipe'))
        self.inventory = 'GPU-free, 81\nGPU-other, 20000\n'
        self.daemons = ('GPU-free, 222, nvidia-cuda-mps-server\n'
                        'GPU-other, 222, nvidia-cuda-mps-server\n')

    def test_permits_known_daemon_when_every_client_maps_to_another_gpu(self):
        applications = self.daemons + 'GPU-other, 333, python\n'
        with patch.object(subprocess, 'check_output', side_effect=[self.inventory, applications]), \
                patch.object(subprocess, 'run', return_value=SimpleNamespace(stdout='222\n333\n', stderr='')):
            self.assertEqual(preflight(self.options), self.inventory)

    def test_refuses_a_selected_gpu_client_or_an_unmapped_client(self):
        for applications in (self.daemons + 'GPU-free, 333, python\n', self.daemons):
            with patch.object(subprocess, 'check_output', side_effect=[self.inventory, applications]), \
                    patch.object(subprocess, 'run', return_value=SimpleNamespace(stdout='222\n333\n', stderr='')):
                with self.assertRaises(RuntimeError):
                    preflight(self.options)


if __name__ == '__main__':
    unittest.main()
