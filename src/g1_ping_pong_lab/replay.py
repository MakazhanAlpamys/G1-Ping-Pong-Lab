"""Play exact saved native states in a local MuJoCo window (no new simulation).

Recorded simulation frames may contain independent trial resets or a continuous
robot with separate new-ball feeds. See the paired report for boundary semantics.
Playback follows the global recording clock; it never infers a same-ball rally.
"""
import argparse
from pathlib import Path
import time

import mujoco
import numpy as np


def load_recording(path, *, nq, nv, nu):
    with np.load(path, allow_pickle=False) as payload:
        fields = {name: payload[name].copy() for name in payload.files}
    clock = fields.get("time")
    if (clock is None or clock.ndim != 1 or len(clock) == 0 or not np.isfinite(clock).all()
        or not np.all(np.diff(clock) > 0)):
        raise ValueError("Recording times must be finite and strictly increasing")
    for name, width in (("qpos", nq), ("qvel", nv), ("ctrl", nu)):
        if fields.get(name, np.empty(0)).shape != (len(clock), width) or not np.isfinite(fields[name]).all():
            raise ValueError("Recording state dimensions or finite values are invalid: " + name)
    return fields


def playback_timing(fields, *, active_seconds, speed):
    """Select frames and use GLOBAL timestamps even when per-feed time restarts."""
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("Playback speed must be finite and positive")
    trial_clock = fields.get("trial_time", fields["time"])
    selected = np.flatnonzero(trial_clock <= active_seconds + 1e-8)
    if len(selected) == 0:
        raise ValueError("No frames selected")
    offsets = (fields["time"][selected] - fields["time"][selected[0]]) / speed
    return selected, offsets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=Path("assets/scene.xml"))
    parser.add_argument("--session", type=Path, default=Path("outputs/demo/session.npz"))
    parser.add_argument("--active-seconds", type=float, default=2.4,
                        help="Include frames within this per-feed/trial interval; use 12 for the full supplied audit")
    parser.add_argument("--speed", type=float, default=1.)
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0 or not .02 <= args.active_seconds <= 12.:
        parser.error("Positive speed and active-seconds in [.02,12] required")
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    fields = load_recording(args.session, nq=model.nq, nv=model.nv, nu=model.nu)
    selected, offsets = playback_timing(fields, active_seconds=args.active_seconds, speed=args.speed)
    data = mujoco.MjData(model)
    from mujoco import viewer as native_viewer
    print("Recorded simulation frames. See the paired report for ball-feed and robot-reset semantics.", flush=True)
    with native_viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [-.7, 0., .85]
        viewer.cam.distance = 4.1
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -23
        viewer.opt.geomgroup[3] = 0
        while viewer.is_running():
            playback_started = time.monotonic()
            for index, offset in zip(selected, offsets):
                if not viewer.is_running():
                    return
                time.sleep(max(0., playback_started + offset - time.monotonic()))
                data.qpos[:] = fields["qpos"][index]
                data.qvel[:] = fields["qvel"][index]
                data.ctrl[:] = fields["ctrl"][index]
                data.time = float(fields["time"][index])
                mujoco.mj_forward(model, data)
                viewer.sync()
            time.sleep(.02 / args.speed)


if __name__ == "__main__":
    main()
