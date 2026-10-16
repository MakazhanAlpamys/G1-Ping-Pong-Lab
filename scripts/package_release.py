"""Build a release with the original demo; keep other video edits local."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {"assets", "data", "docs", "licenses", "media", "scripts", "src", "tests", "tools"}
ROOT_FILES = {"README.md", "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "pyproject.toml", ".gitignore", ".gitattributes"}
SKIP_PARTS = {"__pycache__", "node_modules", ".hyperframes", ".media", ".pytest_cache", "renders"}
LOCAL_FILES = {"docs/edited-videos.md", "licenses/IBMPlexMono-OFL.txt",
               "licenses/LeagueGothic-OFL.txt", "licenses/font-provenance.json"}
MEDIA_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".gif", ".wav", ".mp3"}
PUBLIC_MEDIA = {"media/demo.mp4", "media/demo-preview.gif", "media/poster.png",
                "media/demo.render.json"}


def selected(path):
    relative = path.relative_to(ROOT)
    name = relative.as_posix()
    if name in LOCAL_FILES or (relative.parts[0] == "media" and name not in PUBLIC_MEDIA):
        return False
    if path.suffix.lower() in MEDIA_SUFFIXES and name not in PUBLIC_MEDIA:
        return False
    if path.is_symlink() or any(p in SKIP_PARTS or p.endswith(".egg-info") for p in relative.parts):
        return False
    if path.suffix.lower() in {".mjb", ".pyc", ".log", ".key", ".pem"}:
        return False
    return (len(relative.parts) == 1 and relative.name in ROOT_FILES) or relative.parts[0] in DIRECTORIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "G1-Ping-Pong-Lab-Source.zip")
    args = parser.parse_args()
    target = args.output.resolve()
    if target.is_relative_to(ROOT):
        raise ValueError("Write the release archive outside the project directory")
    candidates = [ROOT / name for name in ROOT_FILES]
    candidates += [p for name in DIRECTORIES for p in (ROOT / name).rglob("*")]
    files = sorted(p for p in candidates if p.is_file() and selected(p))
    patterns = [rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
                rb"\bAKIA[A-Z0-9]{16}\b", rb"\bghp_[A-Za-z0-9]{36}\b",
                rb"\bsk-(?:ant-|proj-)[A-Za-z0-9_-]{20,}",
                rb"(?:api_key|api_token|access_token)\s*[=:]\s*[\"'][A-Za-z0-9_-]{24,}"]
    manifest = {}
    for path in files:
        name = path.relative_to(ROOT).as_posix()
        payload = path.read_bytes()
        if path.name.startswith(".env") or any(re.search(pattern, payload, re.I) for pattern in patterns):
            raise ValueError("Release content review required: " + name)
        if len(payload) >= 100 * 1024 * 1024:
            raise ValueError("File too large for a normal GitHub repository: " + name)
        manifest[name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    required_files = sorted(PUBLIC_MEDIA) + ["docs/validation-report.json", "data/demo-session.npz",
                                           "docs/continuous-report.json", "data/continuous-session.npz"]
    for required in required_files:
        if required not in manifest or manifest[required]["bytes"] == 0:
            raise ValueError("Missing release artifact: " + required)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in manifest:
            archive.write(ROOT / name, name)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(manifest):
            raise ValueError("Release archive inventory mismatch")
        for name, info in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != info["sha256"]:
                raise ValueError("Release readback hash mismatch: " + name)
    receipt = {"archive": target.name, "bytes": target.stat().st_size,
               "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "files": manifest}
    target.with_suffix(".manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"} | {"file_count": len(manifest)}))


if __name__ == "__main__":
    main()
