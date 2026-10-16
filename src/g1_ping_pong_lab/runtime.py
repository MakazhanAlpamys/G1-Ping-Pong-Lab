"""CPU-only native MuJoCo simulation with a frozen reference-residual WBC.

The seven-joint stroke is a CEM-optimized fixed-feed script. The pretrained
132-input / 29-output Apache-2.0 WBC supplies balance. No high-level learned
ball-reactive policy, camera perception, real-hardware deployment or rally is
claimed. A fresh whole trial is the only place where robot/ball state resets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import onnxruntime as ort
import yaml

from .scoring import ReturnAttempt, TableGeometry

FORCE_CAPS = np.array([88., 139., 88., 139., 50., 50., 88., 139., 88., 139., 50., 50., 88., 50., 50.,
                       25., 25., 25., 25., 25., 5., 5., 25., 25., 25., 25., 25., 5., 5.])
ARM_NAMES = ("right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
             "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint")
FIRST = np.array([.9737769961, .8675284982, -.6322987676, .9079850316, .6004983187, -.5216718912, .3581844568,
                  -.9194896221, -.0878282711, .9610280991, .7065397501, -.9304621816, -.3615038693, -.8877418041,
                  .2632637322, .1716955304], dtype=np.float32).astype(np.float64)
ASSET_HASHES = {"policy.onnx": "63a444122d85d2868045e08ff51c3fe711edbabc2d8c96a1a241d4c59e98bb34",
                "config.yaml": "a646687843846d417d24d6b2c9f3a4dc51343d149491f63478363f6e3112fb17"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stroke_at(t):
    """Seven bounded arm-reference offsets, returning to READY after 1.05 s."""
    def smooth(x):
        x = np.clip(x, 0., 1.)
        return x * x * (3. - 2. * x)
    back, forward, start, duration = FIRST[:7], FIRST[7:14], FIRST[14], FIRST[15]
    if t < start:
        return back * smooth((t - .02) / .18)
    if t < .7:
        return np.clip(back + (forward - back) * smooth((t - start) / duration), -1., 1.)
    return np.clip(forward * (1. - smooth((t - .7) / .35)), -1., 1.)


class Controller:
    """Pinned CPU ONNX WBC with pelvis LINK-frame gyro and published raw history."""

    def __init__(self, model, assets):
        self.model = model
        assets = Path(assets)
        for name, expected in ASSET_HASHES.items():
            if sha(assets / name) != expected:
                raise ValueError("WBC asset checksum mismatch: " + name)
        cfg = yaml.safe_load((assets / "config.yaml").read_text())
        self.names = tuple(cfg["joint_names"])
        if len(self.names) != 29 or cfg["action"]["action_mode"] != "reference_residual":
            raise ValueError("Unexpected WBC contract")
        self.default = np.array(cfg["default_joint_pos"])
        self.scale = np.array(cfg["action"]["scale"])
        joint_ids = np.array([model.joint("robot/" + n).id for n in self.names])
        self.qids = model.jnt_qposadr[joint_ids].copy()
        self.vids = model.jnt_dofadr[joint_ids].copy()
        self.ranges = model.jnt_range[joint_ids].copy()
        self.arm = np.array([self.names.index(n) for n in ARM_NAMES])
        ids = [np.flatnonzero(model.actuator_trnid[:, 0] == j) for j in joint_ids]
        if any(len(v) != 1 for v in ids):
            raise ValueError("Every WBC joint needs exactly one native actuator")
        self.actuator_ids = np.array([v[0] for v in ids])
        aid = self.actuator_ids
        if (not np.all(model.actuator_forcelimited[aid])
            or not np.array_equal(model.actuator_forcerange[aid], np.column_stack((-FORCE_CAPS, FORCE_CAPS)))
            or not np.allclose(model.actuator_gainprm[aid, 0], cfg["stiffness"], rtol=1e-5, atol=1e-7)
            or not np.allclose(model.actuator_biasprm[aid, 1], -np.array(cfg["stiffness"]), rtol=1e-5, atol=1e-7)
            or not np.allclose(model.actuator_biasprm[aid, 2], -np.array(cfg["damping"]), rtol=1e-5, atol=1e-7)):
            raise ValueError("Native position-PD or force-cap contract mismatch")
        self.pelvis = model.body("robot/pelvis").id
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(assets / "policy.onnx"), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        if self.session.get_inputs()[0].shape != [1, 132] or self.session.get_outputs()[0].shape != [1, 29]:
            raise ValueError("Expected 132-input / 29-output WBC")
        self.previous = np.zeros(29, dtype=np.float32)

    def reference(self, data):
        torso = self.model.body("robot/torso_link").id
        rotation = data.xmat[torso].reshape(3, 3)
        return np.r_[data.xpos[torso, 2], np.zeros(6), rotation.T @ [0., 0., -1.], data.qpos[self.qids]]

    def apply(self, data, reference):
        mujoco.mj_forward(self.model, data)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, self.pelvis, velocity, 0)
        rotation = data.xmat[self.pelvis].reshape(3, 3)
        observation = np.r_[reference, rotation.T @ velocity[:3], rotation.T @ [0., 0., -1.],
                            data.qpos[self.qids] - self.default, data.qvel[self.vids], self.previous].astype(np.float32)
        raw = self.session.run(None, {self.session.get_inputs()[0].name: observation[None]})[0][0]
        if raw.shape != (29,) or not np.isfinite(raw).all():
            raise ValueError("Non-finite WBC action")
        target = reference[10:] + self.scale * raw
        for i, aid in enumerate(self.actuator_ids):
            if self.model.actuator_ctrllimited[aid]:
                target[i] = np.clip(target[i], *self.model.actuator_ctrlrange[aid])
        data.ctrl[self.actuator_ids] = target
        self.previous = raw.copy()  # Exactly one published-raw29 commit per 50 Hz tick.
        return raw, observation


def run_trial(scene, assets, *, seconds=12., serve_position=(-.65, -.25, 1.), serve_velocity=(-1.8, 0., .5)):
    """Run one physical ball and audit every .25 ms substep through the window."""
    if not np.isfinite(seconds) or seconds <= 0 or abs(seconds / .02 - round(seconds / .02)) > 1e-8:
        raise ValueError("Trial duration must be a positive multiple of .02 seconds")
    model = mujoco.MjModel.from_xml_path(str(scene))
    if abs(model.opt.timestep - .00025) > 1e-12 or model.nmocap or model.neq:
        raise ValueError("Expected free robot, fixed table, no mocap/equality and .25 ms physics")
    controller = Controller(model, assets)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    ball_q = int(model.joint("ball/ball_freejoint").qposadr[0])
    ball_v = int(model.joint("ball/ball_freejoint").dofadr[0])
    data.qpos[ball_q:ball_q + 3] = serve_position
    data.qvel[ball_v:ball_v + 3] = serve_velocity
    mujoco.mj_forward(model, data)
    reference = controller.reference(data)
    ids = {name: model.geom(path).id for name, path in {
        "table": "table/table_top", "net": "table/net", "racket": "robot/racket_collision", "ball": "ball/ball_geom"}.items()}
    center, size = data.geom_xpos[ids["table"]], model.geom_size[ids["table"]]
    geometry = TableGeometry(net_x=float(data.geom_xpos[ids["net"], 0]),
        table_x_min=float(center[0]-size[0]), table_x_max=float(center[0]+size[0]),
        table_y_min=float(center[1]-size[1]), table_y_max=float(center[1]+size[1]),
        net_top_z=float(data.geom_xpos[ids["net"], 2]+model.geom_size[ids["net"], 2]), ball_radius=.02)
    scorer = ReturnAttempt(geometry)
    robot_geoms = set()
    for i in range(model.ngeom):
        ancestor = int(model.geom_bodyid[i])
        while ancestor and ancestor != controller.pelvis:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor == controller.pelvis:
            robot_geoms.add(i)
    table_geoms = {i for i in range(model.ngeom) if model.geom(i).name.startswith("table/")}
    feet = {model.body("robot/" + side + "_ankle_roll_link").id for side in ("left", "right")}
    support_steps = nonfoot_steps = 0
    cap_ratio = 0.
    frames = []
    nsteps = round(seconds / model.opt.timestep)
    failure = None
    for step in range(nsteps):
        if step % 80 == 0:
            ref = reference.copy()
            raw7 = stroke_at(data.time)
            ref[10 + controller.arm] += .7 * raw7
            ref[10:] = np.clip(ref[10:], controller.ranges[:, 0], controller.ranges[:, 1])
            raw29, observation = controller.apply(data, ref)
        mujoco.mj_step(model, data)
        if (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
            or abs(data.time - (step+1)*model.opt.timestep) > 1e-7
            or any(data.warning[k].number for k in (mujoco.mjtWarning.mjWARN_BADQPOS,
                   mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQACC))):
            failure = "physics_warning_or_reset"
            break
        active = set()
        support = nonfoot = False
        for contact in data.contact[:data.ncon]:
            if contact.efc_address < 0 or contact.dist > 0:
                continue
            a, b = int(contact.geom1), int(contact.geom2)
            if (a in robot_geoms and b in table_geoms) or (b in robot_geoms and a in table_geoms):
                support = True
            for ground, other in ((a, b), (b, a)):
                if (model.geom_type[ground] == mujoco.mjtGeom.mjGEOM_PLANE and other in robot_geoms
                    and int(model.geom_bodyid[other]) not in feet):
                    nonfoot = True
            if ids["ball"] in (a, b):
                other = b if a == ids["ball"] else a
                known = {ids[k]: k for k in ("table", "net", "racket")}
                active.add(known.get(other, "floor" if model.geom_type[other] == mujoco.mjtGeom.mjGEOM_PLANE else "body"))
        support_steps += int(support)
        nonfoot_steps += int(nonfoot)
        quat = data.qpos[3:7]
        fallen = data.qpos[2] < .45 or -(1.-2.*(quat[1]**2+quat[2]**2)) > -.5
        scorer.observe(time=data.time, ball_position=data.qpos[ball_q:ball_q + 3], contacts=active, fallen=fallen)
        cap_ratio = max(cap_ratio, float(np.max(abs(data.actuator_force[controller.actuator_ids]) / FORCE_CAPS)))
        if (step + 1) % 80 == 0:
            frames.append({"time": float(data.time), "qpos": data.qpos.copy(), "qvel": data.qvel.copy(),
                           "ctrl": data.ctrl.copy(), "reference39": ref.copy(), "raw7": raw7.copy(),
                           "raw_wbc29": raw29.copy(), "observation132": observation.copy()})
    outcome = scorer.finish(aborted=failure is not None)
    accepted = bool(outcome["success"] and support_steps == nonfoot_steps == 0 and failure is None
                    and len(frames) == round(seconds/.02) and cap_ratio <= 1.000001)
    result = {"ball_outcome": outcome, "accepted_return": accepted, "requested_seconds": seconds,
              "actual_seconds": float(data.time), "physics_substeps": step+1,
              "external_support": bool(support_steps), "external_support_substeps": support_steps,
              "nonfoot_ground_substeps": nonfoot_steps, "maximum_force_cap_ratio": cap_ratio,
              "midtrial_resets": 0, "postlaunch_ball_writes": 0, "failure": failure,
              "serve_position": list(serve_position), "serve_velocity": list(serve_velocity)}
    return result, frames


def record_session(scene, assets, output, *, serves=8):
    if not 1 <= serves <= 100:
        raise ValueError("Serve count must be between 1 and 100")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    reports, states = [], []
    started = time.monotonic()
    for trial in range(serves):
        result, frames = run_trial(scene, assets)
        reports.append({"trial_id": trial, **result})
        states.extend({"time": trial*12. + f["time"], "trial_time": f["time"], "trial_id": trial,
                       **{k: v for k, v in f.items() if k != "time"}} for f in frames)
        print(json.dumps({"trial": trial+1, "accepted_return": result["accepted_return"],
                          "ball_reason": result["ball_outcome"]["reason"], "elapsed_seconds": round(time.monotonic()-started, 2)}), flush=True)
    np.savez_compressed(output / "session.npz", **{k: np.array([f[k] for f in states]) for k in states[0]})
    report = {"protocol": "g1-ping-pong-lab.native-cpu-independent-serves-v1",
              "description": "Identical fixed feeds. Each independent serve resets the entire trial and WBC history. No continuous rally.",
              "controller": "CEM-optimized seven-arm-reference script over frozen Apache-2.0 132/29 WBC",
              "physics_dt": .00025, "control_hz": 50, "trial_seconds": 12., "serves": serves,
              "accepted_returns": sum(r["accepted_return"] for r in reports), "trials": reports,
              "elapsed_wall_seconds": time.monotonic()-started, "mujoco_version": mujoco.__version__,
              "source_sha256": {p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name("scoring.py"))},
              "scene_sha256": sha(scene), "wbc_sha256": ASSET_HASHES, "trajectory_sha256": sha(output / "session.npz")}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=Path("assets/scene.xml"))
    parser.add_argument("--wbc", type=Path, default=Path("assets/wbc"))
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--serves", type=int, default=8)
    args = parser.parse_args()
    record_session(args.scene, args.wbc, args.output, serves=args.serves)


if __name__ == "__main__":
    main()
