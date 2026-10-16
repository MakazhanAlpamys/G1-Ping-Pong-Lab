"""Optional EGL smoke test using the actual portable native scene."""
import importlib.util
import os
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).parents[1]


@unittest.skipUnless(os.environ.get('RUN_RENDER_TESTS') == '1', 'Set RUN_RENDER_TESTS=1 to test the optional EGL renderer')
class NativeFrameTests(unittest.TestCase):
    def test_studio_skybox_does_not_change_compiled_native_physics(self):
        os.environ.setdefault('MUJOCO_GL', 'egl')
        import mujoco
        spec = importlib.util.spec_from_file_location('render_footage', ROOT / 'tools/render_footage.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, 'load_visual_model'), 'Studio skybox loader is missing')
        model = module.load_visual_model(ROOT / 'assets/scene.xml', mujoco)
        self.assertIn(mujoco.mjtTexture.mjTEXTURE_SKYBOX, model.tex_type)

    def test_full_hd_frame_keeps_native_physics_arrays_unchanged(self):
        os.environ.setdefault('MUJOCO_GL', 'egl')
        import mujoco
        spec = importlib.util.spec_from_file_location('render_footage', ROOT / 'tools/render_footage.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = mujoco.MjModel.from_xml_path(str(ROOT / 'assets/scene.xml'))
        original = {key: np.array(getattr(model, key), copy=True)
                    for key in ['geom_pos', 'geom_size', 'geom_contype', 'geom_conaffinity',
                                'actuator_forcerange', 'dof_damping', 'body_mass']}
        module.decorate_model(model, mujoco)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        with mujoco.Renderer(model, height=1080, width=1920) as renderer:
            renderer.update_scene(data, camera=module.camera_for(mujoco, 'wide', 0.))
            module.decorate_scene(renderer.scene, mujoco)
            image = renderer.render()
        self.assertEqual(image.shape, (1080, 1920, 3))
        self.assertGreater(float(np.std(image)), 10.)
        for key, value in original.items():
            np.testing.assert_array_equal(getattr(model, key), value, err_msg=key)


if __name__ == '__main__':
    unittest.main()
