from pathlib import Path
import runpy
import unittest
from unittest.mock import patch


script = Path(__file__).resolve().parents[1] / 'scripts' / 'schedule_k3_cutoff.py'
cutoff = runpy.run_path(str(script))


class CutoffTests(unittest.TestCase):
    def test_tracks_session_and_detached_descendant_without_matching_reused_pid(self):
        Process = cutoff['Process']
        running = {
            20: Process(1, 20, 100),
            21: Process(1, 20, 101),
            22: Process(21, 22, 102),
            23: Process(1, 23, 103),
            80: Process(1, 80, 200),
        }
        known = {20: 100, 80: 120}
        collect = cutoff['owned_processes']
        with patch.dict(collect.__globals__, {'process': running.get}), \
             patch.object(Path, 'iterdir', side_effect=lambda: map(lambda pid: Path(str(pid)), running)):
            self.assertEqual(set(collect(known)), {20, 21, 22})
            del running[20]
            running[22] = Process(1, 22, 102)
            self.assertEqual(set(collect(known)), {21, 22})
