import importlib.util
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).parents[1] / 'tools/render_continuous.py'


class ContinuousFrameTests(unittest.TestCase):
    def load(self):
        self.assertTrue(SCRIPT.exists(), 'Continuous renderer is not implemented')
        spec = importlib.util.spec_from_file_location('render_continuous', SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_global_time_is_not_restarted_at_a_new_feed(self):
        module = self.load()
        times = np.array([.02, .04, .06, .08, .10])
        self.assertEqual(module.global_frame_index(times, .075), 2)
        self.assertEqual(module.global_frame_index(times, .10), 4)
        with self.assertRaises(ValueError):
            module.global_frame_index(np.array([.02, .04, .02]), .03)

    def test_plan_uses_every_elapsed_second_without_montage_cuts(self):
        module = self.load()
        arrays = {'time': np.array([.02, .04, .06, .08, .10, .12]),
                  'trial_id': np.array([0, 0, 1, 1, 2, 2])}
        plan = module.continuous_plan(arrays, seconds=.1, fps=30)
        self.assertEqual([item['recorded_index'] for item in plan], [0, 1, 3])
        self.assertEqual([item['video_frame'] for item in plan], [0, 1, 2])
        self.assertFalse(any(item['cut_before'] for item in plan[1:]))

    def test_continuous_film_refuses_robot_or_balance_history_resets(self):
        module = self.load()
        self.assertTrue(hasattr(module, 'validate_continuity_report'), 'Continuity-report gate is missing')
        module.validate_continuity_report({'robot_resets_after_start': 0, 'wbc_history_resets_after_start': 0})
        with self.assertRaises(ValueError):
            module.validate_continuity_report({'robot_resets_after_start': 1, 'wbc_history_resets_after_start': 0})
        with self.assertRaises(ValueError):
            module.validate_continuity_report({'robot_resets_after_start': 0, 'wbc_history_resets_after_start': 1})
        with self.assertRaises(ValueError):
            module.validate_continuity_report({})

    def test_recording_and_scene_must_match_the_continuity_report(self):
        module = self.load()
        self.assertTrue(hasattr(module, 'validate_input_provenance'), 'Recorded-input provenance gate is missing')
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / 'scene.xml'
            states = Path(directory) / 'states.npz'
            scene.write_bytes(b'scene fixture')
            states.write_bytes(b'physical recording fixture')
            report = {'scene_sha256': hashlib.sha256(scene.read_bytes()).hexdigest(),
                      'trajectory_sha256': hashlib.sha256(states.read_bytes()).hexdigest()}
            module.validate_input_provenance(report, scene, states)
            states.write_bytes(b'a different recording')
            with self.assertRaises(ValueError):
                module.validate_input_provenance(report, scene, states)


if __name__ == '__main__':
    unittest.main()
