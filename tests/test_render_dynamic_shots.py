import importlib.util
from pathlib import Path
import unittest

import numpy as np

SCRIPT = Path(__file__).parents[1] / 'tools/render_dynamic_shots.py'


class DynamicShotTests(unittest.TestCase):
    def load(self):
        self.assertTrue(SCRIPT.exists(), 'Dynamic shot renderer is not implemented')
        spec = importlib.util.spec_from_file_location('render_dynamic_shots', SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def fixture(self):
        return {'trial_id': np.repeat([0, 1], 10),
                'trial_time': np.tile(np.arange(10) * .02 + .10, 2)}

    def test_fast_feed_shot_selects_real_states_from_only_its_own_trial(self):
        module = self.load()
        frames = module.dynamic_plan(self.fixture(), trial=1, start=.10, seconds=.1, fps=30, speed=1.)
        self.assertEqual([frame['recorded_index'] for frame in frames], [10, 11, 13])
        self.assertTrue(all(frame['trial_id'] == 1 for frame in frames))

    def test_slow_motion_repeats_saved_states_without_inventing_intermediate_states(self):
        module = self.load()
        frames = module.dynamic_plan(self.fixture(), trial=0, start=.10, seconds=.1, fps=30, speed=.5)
        self.assertEqual([frame['recorded_index'] for frame in frames], [0, 0, 1])
        with self.assertRaises(ValueError):
            module.dynamic_plan(self.fixture(), trial=0, start=.25, seconds=1., fps=30, speed=1.)

    def test_replay_provenance_uses_the_camera_that_is_actually_rendered(self):
        module = self.load()
        self.assertTrue(hasattr(module, 'camera_parameters'), 'Actual camera metadata factory is missing')
        import mujoco
        model = mujoco.MjModel.from_xml_string('<mujoco/>')
        parameters = module.camera_parameters('portrait', 4, replay=True)
        camera = module.camera_for(mujoco, model, 'portrait', 4, 0., replay=True)
        self.assertEqual(camera.distance, parameters[2])
        self.assertEqual(camera.azimuth, parameters[0])
        self.assertEqual(camera.elevation, parameters[1])
        self.assertAlmostEqual(model.vis.global_.fovy, parameters[3])
        np.testing.assert_array_equal(camera.lookat, parameters[4])


if __name__ == '__main__':
    unittest.main()
