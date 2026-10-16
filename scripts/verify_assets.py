"""Verify required portable assets; no downloads, cloud access or binary model needed."""
import argparse
import hashlib
import json
from pathlib import Path


def verify_assets(directory):
    root = Path(directory).resolve()
    manifest = json.loads((root / "model-manifest.json").read_text())
    required = manifest.get("sha256", {})
    if not isinstance(required, dict) or not required:
        raise ValueError("Required asset inventory is missing")
    for name, expected in required.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Required asset is missing or outside the asset directory: " + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Required asset checksum mismatch: " + name)
    return len(required)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path(__file__).resolve().parents[1] / "assets")
    args = parser.parse_args()
    count = verify_assets(args.assets)
    print(f"Verified {count} required portable assets. Compiled MJB is not required.")


if __name__ == "__main__":
    main()
