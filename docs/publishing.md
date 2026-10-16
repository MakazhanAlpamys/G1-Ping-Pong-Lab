# Publishing the repository

The source repository's remote is
[MakazhanAlpamys/G1-Ping-Pong-Lab](https://github.com/MakazhanAlpamys/G1-Ping-Pong-Lab).
Push only the intended source files and the original showcase media with Git.
The published media allowlist is `media/demo.mp4`, `media/demo-preview.gif`,
`media/poster.png` and `media/demo.render.json`. The README links to this original
six-second uncut demo. Additional renders and editing projects remain local.

Suggested repository name: **g1-ping-pong-lab**

Suggested description:

> A reproducible Unitree G1 table-tennis simulation demo with native physics validation.

Suggested topics: `robotics`, `unitree-g1`, `mujoco`, `table-tennis`, `simulation`, `onnx`.

## With Git

Run from this project's root for the initial upload.

```bash
git init
git add .
git commit -m "Release G1 Ping Pong Lab demo"
git branch -M main
git remote add origin https://github.com/MakazhanAlpamys/G1-Ping-Pong-Lab.git
git push -u origin main
```

Before committing, inspect `git status --short` and `git diff --cached --stat`.
Keep additional generated media and local editing files ignored; do not
force-add files outside the published media allowlist.

Keep `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`, the upstream license files and
asset provenance in the uploaded repository. They cover the included pretrained
policy, robot description and meshes.

Generated trajectories, additional media, caches and build folders are ignored.
The compact validation reports in `docs/` and the frozen simulation state files
in `data/` remain included so the physical results can be inspected and reproduced.
