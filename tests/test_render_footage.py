import importlib.util
from pathlib import Path
import unittest

import numpy as np

SCRIPT = Path(__file__).parents[1] / 'tools' / 'render_footage.py'


class RecordedFrameTests(unittest.TestCase):
    def load(self):
        self.assertTrue(SCRIPT.exists(), 'Offline renderer is not implemented yet')
        spec = importlib.util.spec_from_file_location('render_footage', SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_sampling_uses_an_actual_saved_state_in_requested_serve(self):
        module = self.load()
        arrays = {'trial_id': np.array([0, 0, 0, 1, 1, 1]),
                  'trial_time': np.array([.02, .04, .06, .02, .04, .06])}
        self.assertEqual(module.recorded_frame_index(arrays, 1, .039), 3)
        self.assertEqual(module.recorded_frame_index(arrays, 0, .06), 2)
        self.assertEqual(module.recorded_frame_index(arrays, 0, 4.), 2)
        with self.assertRaises(ValueError):
            module.recorded_frame_index(arrays, 2, .04)

    def test_malformed_recording_is_rejected_before_rendering(self):
        module = self.load()
        arrays = {'trial_id': np.array([0, 0]), 'trial_time': np.array([.04, .02])}
        with self.assertRaises(ValueError):
            module.recorded_frame_index(arrays, 0, .03)


if __name__ == '__main__':
    unittest.main()
