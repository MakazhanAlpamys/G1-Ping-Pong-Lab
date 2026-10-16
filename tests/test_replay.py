import importlib
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


class ReplayTests(unittest.TestCase):
    def test_global_replay_clock_does_not_restart_when_feed_time_resets(self):
        replay = importlib.import_module("g1_ping_pong_lab.replay")
        self.assertTrue(hasattr(replay, "playback_timing"), "Global playback scheduling is not implemented")
        fields = {"time": np.array([2.98, 3., 3.02, 3.04]),
                  "trial_time": np.array([2.98, 3., .02, .04])}
        indices, offsets = replay.playback_timing(fields, active_seconds=12., speed=1.)
        np.testing.assert_array_equal(indices, [0, 1, 2, 3])
        np.testing.assert_allclose(offsets, [0., .02, .04, .06], atol=1e-12)

    def test_cli_validates_recording_before_opening_window(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.npz"
            np.savez(path, time=[.02, .01], qpos=np.zeros((2, 43)), qvel=np.zeros((2, 41)), ctrl=np.zeros((2, 29)))
            root = Path(__file__).resolve().parents[1]
            result = subprocess.run([sys.executable, "-m", "g1_ping_pong_lab.replay", "--scene",
                                     str(root / "assets/scene.xml"), "--session", str(path)],
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Recording times must be finite and strictly increasing", result.stderr)

    def test_invalid_or_unordered_recordings_are_rejected(self):
        try:
            replay = importlib.import_module("g1_ping_pong_lab.replay")
        except ImportError as error:
            self.fail("Native replay has not been implemented: " + str(error))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.npz"
            np.savez(path, time=[.02, .01], qpos=np.zeros((2, 43)), qvel=np.zeros((2, 41)), ctrl=np.zeros((2, 29)))
            with self.assertRaises(ValueError):
                replay.load_recording(path, nq=43, nv=41, nu=29)


if __name__ == "__main__":
    unittest.main()
