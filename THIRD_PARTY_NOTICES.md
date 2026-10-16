# Third-party notices

The project's original code, documentation, stroke parameters, recorded
simulation states and original demo video use [Apache-2.0](LICENSE). Third-party material keeps
its own license and attribution. A license for the simulator is not a claim
that the controller has been transferred to a physical robot.

## Included controller assets

`assets/wbc/policy.onnx` and `assets/wbc/config.yaml` are unmodified files from
[wbc-mjlab/wbc-g1-deploy](https://github.com/wbc-mjlab/wbc-g1-deploy/tree/6dabf86fddc2b7b429b09e74999732fcde3441f9),
commit `6dabf86fddc2b7b429b09e74999732fcde3441f9`.

| Asset | SHA-256 |
| --- | --- |
| `policy.onnx` | `63a444122d85d2868045e08ff51c3fe711edbabc2d8c96a1a241d4c59e98bb34` |
| `config.yaml` | `a646687843846d417d24d6b2c9f3a4dc51343d149491f63478363f6e3112fb17` |

These assets are covered by the upstream repository's Apache-2.0 license.
The complete [license](licenses/WBC-LICENSE.txt) and
[upstream NOTICE](licenses/WBC-NOTICE.txt) are included. This project uses
the pretrained controller for balance and body coordination; it did not train
that controller. Its scripted stroke parameters are a separate contribution.

## Included robot description and meshes

The robot part of `assets/scene.xml` was exported and modified from
[mjlab's G1 description](https://github.com/mujocolab/mjlab/blob/0fb8a681136be94ffc636a3dd423cabb97d91f10/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml),
version 1.6.0, commit `0fb8a681136be94ffc636a3dd423cabb97d91f10`.
Copyright 2025, The mjlab Developers. The
[mjlab Apache-2.0 license](licenses/mjlab-LICENSE.txt) is included.

Project changes comprise the exported scene structure, table, net, ball,
paddle, stance and actuator configuration, and removal of the rebound wall.
See the scene's modification comment and the runtime documentation for the
current configuration.

The 35 STL files in `assets/meshes/` are Unitree G1 geometry under BSD-3-Clause.
Each was compared byte for byte with the
[official Unitree repository](https://github.com/unitreerobotics/unitree_ros/tree/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/g1_description/meshes)
at commit `5994d4faef0a9cadd3287f8de0199a67eeb2a259`.
The [complete Unitree license](licenses/Unitree-BSD-3-Clause.txt) is retained,
including the non-endorsement condition and warranty disclaimer.
The file-by-file source URLs and hashes are in
[asset-provenance.json](licenses/asset-provenance.json).

The [MuJoCo Menagerie G1 description](https://github.com/google-deepmind/mujoco_menagerie/tree/62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35/unitree_g1)
also documents the model's Unitree origin and BSD-3-Clause license. That link
is background provenance; the included STL bytes were verified against
Unitree's own repository.

## Software dependencies

Dependencies are installed separately, rather than copied into this repository.
Their distributions contain their own notices and bundled-component licenses.

| Project | Use | Upstream license |
| --- | --- | --- |
| [MuJoCo 3.11.0](https://github.com/google-deepmind/mujoco/tree/3.11.0) | Native rigid-body physics and rendering | [Apache-2.0](licenses/MuJoCo-LICENSE.txt) |
| [ONNX Runtime 1.31.0](https://github.com/microsoft/onnxruntime/tree/v1.31.0) | Frozen controller inference | [MIT](licenses/ONNX-Runtime-LICENSE.txt) |
| [NumPy](https://github.com/numpy/numpy) | Numeric arrays and trajectory data | BSD-3-Clause for NumPy itself; distribution includes additional notices |
| [PyYAML](https://github.com/yaml/pyyaml) | Controller configuration | MIT |
| [Pillow](https://github.com/python-pillow/Pillow) | Video graphics and image composition | MIT-CMU |

If redistributing a packaged Python environment, simulator binary, browser
client, or FFmpeg executable, retain that distribution's additional license
files as well. This repository does not redistribute those binaries.

## Names and affiliation

Unitree and G1 identify the robot being simulated. The upstream authors retain
their respective rights. This independent demonstration does not imply their
endorsement, and the project license grants no trademark rights.
