import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts/package_release.py"
spec = importlib.util.spec_from_file_location("package_release", SCRIPT)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class LocalMediaTests(unittest.TestCase):
    def test_source_release_excludes_local_clips_and_edit_assets(self):
        for relative in ("media/demo2.mp4", "media/demo-reels.mp4",
                         "media/demo2-preview.gif", "media/full-session.mp4",
                         "media/demo-reels-poster.png",
                         "edits/studio/assets/audio/contact-sfx.wav",
                         "docs/accidental-video.mp4"):
            with self.subTest(path=relative):
                self.assertFalse(release.selected(release.ROOT / relative))

    def test_release_retains_only_the_original_demo_and_its_preview_assets(self):
        for relative in ("media/demo.mp4", "media/demo-preview.gif",
                         "media/poster.png", "media/demo.render.json"):
            with self.subTest(path=relative):
                self.assertTrue(release.selected(release.ROOT / relative))

    def test_source_release_retains_runtime_and_validation_data(self):
        for relative in ("src/g1_ping_pong_lab/controller.py",
                         "assets/wbc/policy.onnx", "data/demo-session.npz",
                         "docs/validation-report.json", "README.md"):
            with self.subTest(path=relative):
                self.assertTrue(release.selected(release.ROOT / relative))


if __name__ == "__main__":
    unittest.main()
