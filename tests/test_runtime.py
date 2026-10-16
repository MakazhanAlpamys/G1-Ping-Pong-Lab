"""Behavior checks for honest independent serves and the native controller."""
import importlib
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def feature(module):
    try:
        return importlib.import_module("g1_ping_pong_lab." + module)
    except ImportError as error:
        raise AssertionError("Standalone runtime feature has not been implemented: " + module) from error


class ScoringTests(unittest.TestCase):
    def test_actual_own_bounce_hit_net_and_far_bounce_are_required(self):
        scoring = feature("scoring")
        attempt = scoring.ReturnAttempt(scoring.TableGeometry())
        for t, p, contacts in [(.1, (-1., 0., .78), {"table"}),
                               (.2, (-1.3, 0., .85), {"racket"}),
                               (.3, (-.1, 0., 1.1), set()),
                               (.4, (.1, 0., 1.1), set()),
                               (.5, (.8, 0., .78), {"table"})]:
            attempt.observe(time=t, ball_position=p, contacts=contacts)
        self.assertTrue(attempt.finish()["success"])

    def test_pretty_flight_without_contact_cannot_score(self):
        scoring = feature("scoring")
        attempt = scoring.ReturnAttempt(scoring.TableGeometry())
        for t, p in [(.1, (-1., 0., .78)), (.2, (-1.3, 0., .85)),
                     (.3, (-.1, 0., 1.1)), (.4, (.1, 0., 1.1)), (.5, (.8, 0., .78))]:
            attempt.observe(time=t, ball_position=p)
        self.assertFalse(attempt.finish()["success"])

    def test_late_fall_invalidates_good_return(self):
        scoring = feature("scoring")
        attempt = scoring.ReturnAttempt(scoring.TableGeometry())
        for t, p, c in [(.1, (-1., 0., .78), {"table"}), (.2, (-1.3, 0., .9), {"racket"}),
                        (.3, (-.1, 0., 1.1), set()), (.4, (.1, 0., 1.1), set()),
                        (.5, (.8, 0., .78), {"table"})]:
            attempt.observe(time=t, ball_position=p, contacts=c)
        attempt.observe(time=10., ball_position=(3., 0., .1), fallen=True)
        self.assertFalse(attempt.finish()["success"])


class ControllerTests(unittest.TestCase):
    def test_stroke_recovers_without_unbounded_reference(self):
        runtime = feature("runtime")
        values = np.array([runtime.stroke_at(t) for t in np.linspace(0., 12., 601)])
        self.assertLessEqual(float(np.max(abs(values))), 1.)
        np.testing.assert_array_equal(values[0], np.zeros(7))
        np.testing.assert_array_equal(values[-1], np.zeros(7))

    def test_prepared_scene_really_has_no_wall_and_keeps_native_caps(self):
        import mujoco
        runtime = feature("runtime")
        model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/scene.xml"))
        self.assertFalse(any("wall" in model.geom(i).name.lower() for i in range(model.ngeom)))
        self.assertFalse(any("wall" in model.body(i).name.lower() for i in range(model.nbody)))
        controller = runtime.Controller(model, ROOT / "assets/wbc")
        np.testing.assert_array_equal(model.actuator_forcerange[controller.actuator_ids],
                                     np.column_stack((-runtime.FORCE_CAPS, runtime.FORCE_CAPS)))
        self.assertEqual(model.nmocap, 0)
        self.assertEqual(model.neq, 0)

    def test_native_trial_counts_full_window_and_history_without_respawning(self):
        runtime = feature("runtime")
        result, frames = runtime.run_trial(ROOT / "assets/scene.xml", ROOT / "assets/wbc", seconds=1.2)
        self.assertTrue(result["ball_outcome"]["success"])
        self.assertEqual(result["physics_substeps"], 4800)
        self.assertEqual(result["midtrial_resets"], 0)
        self.assertEqual(result["postlaunch_ball_writes"], 0)
        self.assertFalse(result["external_support"])
        self.assertEqual(result["nonfoot_ground_substeps"], 0)
        self.assertLessEqual(result["maximum_force_cap_ratio"], 1.000001)
        self.assertEqual(len(frames), 60)
        self.assertTrue(np.isfinite([row["qpos"] for row in frames]).all())


if __name__ == "__main__":
    unittest.main()
