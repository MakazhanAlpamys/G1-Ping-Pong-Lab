import importlib
from pathlib import Path
import unittest

import numpy as np


class ContinuousFeedsTests(unittest.TestCase):
    def test_robot_relative_feeder_rotates_only_new_launch_state(self):
        module = importlib.import_module("g1_ping_pong_lab.continuous_feeds")
        self.assertTrue(hasattr(module, "relative_feed_state"), "Robot-relative feeder has not been implemented")
        initial = np.array([-1.8, 0., .76, 1., 0., 0., 0.])
        current = np.array([-1.9, .1, .76, np.sqrt(.5), 0., 0., np.sqrt(.5)])
        position, velocity = module.relative_feed_state(initial, current)
        np.testing.assert_allclose(position, [-1.65, 1.25, 1.], atol=1e-12)
        np.testing.assert_allclose(velocity, [0., -1.8, .5], atol=1e-12)

    def test_robot_state_and_wbc_history_are_preserved_between_ball_feeds(self):
        try:
            module = importlib.import_module("g1_ping_pong_lab.continuous_feeds")
        except ImportError as error:
            self.fail("Continuous physical feed runner has not been implemented: " + str(error))
        root = Path(__file__).resolve().parents[1]
        report, frames = module.run_continuous(root / "assets/scene.xml", root / "assets/wbc",
                                                feeds=2, interval=3., tail=1.2)
        self.assertEqual(report["robot_resets_after_start"], 0)
        self.assertEqual(report["wbc_history_resets_after_start"], 0)
        self.assertEqual(report["ball_replacements_after_start"], 1)
        self.assertEqual(report["physics_substeps"], 16800)
        self.assertTrue(all(r["robot_qpos_change_at_feed"] == 0. for r in report["feed_boundaries"]))
        for previous, current in zip(frames, frames[1:]):
            np.testing.assert_array_equal(current["observation132"][-29:], previous["raw_wbc29"])
        self.assertGreater(float(np.linalg.norm(frames[150]["observation132"][-29:])), .01)


if __name__ == "__main__":
    unittest.main()
