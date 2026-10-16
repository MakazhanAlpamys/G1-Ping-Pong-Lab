import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


class AssetVerificationTests(unittest.TestCase):
    def verifier(self):
        path = Path(__file__).resolve().parents[1] / "scripts/verify_assets.py"
        if not path.is_file():
            self.fail("Portable shipped-asset verification has not been implemented")
        spec = importlib.util.spec_from_file_location("verify_assets", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.verify_assets

    def test_required_assets_are_checked_and_optional_binary_not_required(self):
        verify = self.verifier()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "scene.xml").write_bytes(b"hello")
            manifest = {"sha256": {"scene.xml": hashlib.sha256(b"hello").hexdigest()},
                        "optional_archive_artifacts": {"scene.mjb": {"required": False}}}
            (root / "model-manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(verify(root), 1)
            (root / "scene.xml").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify(root)

    def test_missing_required_asset_is_rejected(self):
        verify = self.verifier()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "model-manifest.json").write_text(json.dumps({"sha256": {"scene.xml": "0"*64}}))
            with self.assertRaises(ValueError):
                verify(root)


if __name__ == "__main__":
    unittest.main()
