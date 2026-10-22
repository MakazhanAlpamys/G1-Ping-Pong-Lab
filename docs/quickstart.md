# Quick start

Run commands from the repository root. The uncut reference demo is one uninterrupted,
real-time view of **two scheduled feeds to the same continuously simulated
robot**. The feeder replaces the ball at 3 seconds; robot state and balance
controller history are preserved. This is a practice feeder, not a same-ball rally.

For the separate predictive-timing sensing experiment, see the
[ball measurement noise and latency benchmark](noise-latency.md).

## Install and check

The reference environment is Python 3.12, Ubuntu / WSL, MuJoCo 3.11.0 and CPU
ONNX inference. The physical simulation does not require a training GPU.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test,render]"
python -m pytest -q
```

On Windows PowerShell, create the environment with `py -3.12 -m venv .venv`
and activate with `.\.venv\Scripts\Activate.ps1`. The offscreen rendering
commands below use Bash and were verified under Ubuntu / WSL with EGL.

## Run the continuous feeder

```bash
python -m g1_ping_pong_lab.continuous_feeds --feeds 2 --interval 3 --tail 12 --output outputs/continuous-demo
```

The feeds launch at 0 and 3 seconds. The robot is initialized once. The simulator
continues for 12 seconds after the second launch, producing a **15-second**
physical observation window. At each feed boundary only the ball is replaced;
after launch it is simulated normally until the next scheduled feed.

The command writes `session.npz` and `report.json` into the requested directory.
That directory must not already exist; choose another output name for a new run.
The release also supplies the verified recording in
`data/continuous-session.npz` and its report in `docs/continuous-report.json`.

The two-feed result does not establish reliable longer play. Four-feed probes
returned 2/4 at both 3-second and 4-second intervals. See
[results and limitations](results.md) before increasing the feed count.

To open the complete continuous recording in a native MuJoCo window:

```bash
g1-ping-pong-replay --session data/continuous-session.npz --active-seconds 12
```

Here `12` includes each feed's full recorded interval; playback follows the
global clock and includes all 15 seconds. A native window needs a working
display, such as WSLg on WSL. Close the window to stop.

## Rebuild the uncut video

The [original six-second demo](../media/demo.mp4) and its
[render receipt](../media/demo.render.json) are included. The commands below
rebuild that video and generate a full-length audit video locally.

Install **FFmpeg** and place `ffmpeg` or `ffmpeg.exe` on `PATH`. Offscreen
rendering needs EGL/OpenGL libraries and a compatible graphics driver; these
are separate from the CPU physics requirements. For supported graphics
backends, see the official
[MuJoCo Python documentation](https://mujoco.readthedocs.io/en/stable/python.html).

```bash
export MUJOCO_GL=egl
python tools/render_continuous.py --recording data/continuous-session.npz --report docs/continuous-report.json --seconds 6 --output outputs/videos/demo.mp4
python tools/render_continuous.py --recording data/continuous-session.npz --report docs/continuous-report.json --seconds 15 --output outputs/videos/full-session.mp4
```

For a freshly generated run, replace the recording and report arguments with
`outputs/continuous-demo/session.npz` and `outputs/continuous-demo/report.json`.

The main MP4 shows the first six seconds at **1x speed**. The full-session MP4
shows all 15 seconds, including the stance observation tail. Each uses a single
camera moving smoothly through the scene: no cuts, angle changes, slow motion,
titles or music. The ball replacement at 3 seconds is an explicit source
simulation event, not a video edit.

The renderer selects exact saved states; it does not interpolate a strike or
run new physics. A `.render.json` receipt records source hashes, feed times,
playback speed and the physical timestamp of each frame. Studio lighting,
colors and render-only table/net detail affect appearance only.

For FFmpeg outside `PATH`, append `--ffmpeg "/path/to/ffmpeg"` to a renderer
command. You can set `FFMPEG_BIN` and pass `--ffmpeg "$FFMPEG_BIN"` explicitly;
the renderer does not read that variable automatically.

The optional full-HD graphics checks are:

```bash
MUJOCO_GL=egl RUN_RENDER_TESTS=1 python -m pytest -q tests/test_render_native_frame.py
```

## Rebuild the uncut preview

The included [animated preview](../media/demo-preview.gif) and
[poster](../media/poster.png) can be rebuilt from the same uncut main video:

```bash
"${FFMPEG_BIN:-ffmpeg}" -y -i outputs/videos/demo.mp4 -t 6 -vf "fps=15,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3" -loop 0 outputs/videos/demo-preview.gif
"${FFMPEG_BIN:-ffmpeg}" -y -ss 0.5 -i outputs/videos/demo.mp4 -frames:v 1 outputs/videos/poster.png
```

The repository publishes only the original showcase media listed in the
[publishing guide](publishing.md). Keep additional renders local.

## Separate baseline: eight reset trials

The earlier baseline is included separately as `data/demo-session.npz` with
[validation-report.json](validation-report.json) and
[validation-verification.json](validation-verification.json). It contains eight
identical **independent trials**, each resetting the robot, ball and WBC history.
It must not be presented as eight consecutive returns by a persistent robot.

To rerun that baseline:

```bash
g1-ping-pong --serves 8 --output outputs/demo
```

The reference baseline took approximately 147 seconds of wall time for eight
12-second trials. Runtime depends on the machine. The output folder must not
already exist.

To open its native replay, use a working display; WSL requires WSLg or an
alternative display setup:

```bash
g1-ping-pong-replay --session data/demo-session.npz --active-seconds 2.4
g1-ping-pong-replay --session data/demo-session.npz --active-seconds 12
```

The first command selects the baseline stroke windows; the second includes each
full 12-second trial. Playback preserves the global clock, including gaps
between selected windows. Both replay recorded states rather than evaluate a
new policy. Close the viewer window to stop.

## Sources and licenses

The frozen balance network is from
[wbc-mjlab/wbc-g1-deploy](https://github.com/wbc-mjlab/wbc-g1-deploy/tree/6dabf86fddc2b7b429b09e74999732fcde3441f9).
The scene derives from
[mjlab 1.6.0](https://github.com/mujocolab/mjlab/tree/0fb8a681136be94ffc636a3dd423cabb97d91f10/src/mjlab/asset_zoo/robots/unitree_g1),
with meshes verified against
[official Unitree files](https://github.com/unitreerobotics/unitree_ros/tree/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/g1_description/meshes).
See [third-party notices](../THIRD_PARTY_NOTICES.md) for licenses and hashes.
