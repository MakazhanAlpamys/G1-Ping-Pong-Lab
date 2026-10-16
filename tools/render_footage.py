"""Render exact recorded MuJoCo states; cameras and scene decoration are visual only.

This tool never integrates physics or interpolates robot/ball trajectories. Each
output frame selects one saved state from one independently initialized serve.
The accompanying JSON lists those source states and all edits to the appearance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np


def recorded_frame_index(arrays, trial: int, timestamp: float) -> int:
    ids = np.flatnonzero(np.asarray(arrays['trial_id']) == trial)
    if not len(ids):
        raise ValueError(f'No recorded serve {trial}')
    times = np.asarray(arrays['trial_time'])[ids]
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Saved times within a serve must increase strictly')
    return int(ids[np.clip(np.searchsorted(times, timestamp, side='right') - 1, 0, len(ids) - 1)])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def windows_destination(path: Path, executable: str) -> str:
    value = str(path.resolve())
    if executable.lower().endswith('.exe') and value.startswith('/mnt/') and len(value) > 7:
        return value[5].upper() + ':/' + value[7:]
    return value


def find_ffmpeg(explicit=None):
    if explicit:
        return explicit
    found = shutil.which('ffmpeg') or shutil.which('ffmpeg.exe')
    if found:
        return found
    raise RuntimeError('ffmpeg is required; pass its path with --ffmpeg')


def camera_for(mujoco, shot: str, progress: float):
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    if shot == 'close':
        camera.lookat[:] = [-1.42, -.19, .97]
        camera.distance = 2.20
        camera.azimuth = 120. + progress * 16.
        camera.elevation = -12.
    elif shot == 'orbit':
        camera.lookat[:] = [-.56, 0., .70]
        camera.distance = 4.25
        camera.azimuth = 138. - progress * 60.
        camera.elevation = -18. + 4. * progress
    else:
        camera.lookat[:] = [-.37, 0., .65]
        camera.distance = 4.35
        camera.azimuth = 132. - 6. * progress
        camera.elevation = -20.
    return camera


def load_visual_model(path: Path, mujoco):
    if path.suffix == '.mjb':
        return mujoco.MjModel.from_binary_path(str(path))
    native = mujoco.MjModel.from_xml_path(str(path))
    spec = mujoco.MjSpec.from_file(str(path))
    spec.add_texture(name='studio_sky', type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                     rgb1=[.075, .095, .12], rgb2=[.075, .095, .12], width=64, height=384)
    visual = spec.compile()
    for key in ['qpos0', 'body_pos', 'body_quat', 'body_mass', 'body_inertia',
                'geom_bodyid', 'geom_pos', 'geom_quat', 'geom_size', 'geom_contype',
                'geom_conaffinity', 'geom_solref', 'geom_solimp', 'geom_friction',
                'dof_damping', 'dof_armature', 'jnt_range', 'actuator_forcerange',
                'actuator_ctrlrange', 'actuator_gainprm', 'actuator_biasprm']:
        if not np.array_equal(getattr(native, key), getattr(visual, key)):
            raise ValueError('Visual skybox compilation changed native physics: ' + key)
    if (native.nq, native.nv, native.nu, native.opt.timestep) != (visual.nq, visual.nv, visual.nu, visual.opt.timestep):
        raise ValueError('Visual model has a different native control contract')
    return visual


def decorate_model(model, mujoco):
    model.vis.global_.offwidth = 1920
    model.vis.global_.offheight = 1080
    model.vis.global_.fovy = 40.
    model.vis.quality.shadowsize = 8192
    model.vis.map.shadowclip = 2.
    if model.nlight:
        model.light_dir[0] = [.4, .6, -1.]
    model.vis.headlight.ambient[:] = [.55, .55, .57]
    model.vis.headlight.diffuse[:] = [.65, .65, .65]
    model.vis.headlight.specular[:] = [.2, .2, .2]
    model.vis.rgba.haze[:] = [.075, .095, .12, 1.]
    model.vis.map.znear = .01
    changes = []
    for index in range(model.nmat):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, index) or ''
        if 'silver' in name:
            model.mat_rgba[index] = [.88, .91, .92, 1.]
            model.mat_shininess[index] = .35
            changes.append([name, 'bright satin robot finish'])
    for index in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ''
        if 'table_top' in name:
            model.geom_rgba[index] = [.045, .30, .36, 1.]
            model.geom_matid[index] = -1
            changes.append([name, 'teal finish'])
        elif model.geom_type[index] == mujoco.mjtGeom.mjGEOM_PLANE:
            model.geom_rgba[index] = [.075, .095, .12, 1.]
            model.geom_matid[index] = -1
            changes.append([name, 'charcoal studio floor'])
        elif 'ball_geom' in name:
            model.geom_rgba[index] = [1., .41, .025, 1.]
            changes.append([name, 'orange finish, unchanged physical radius'])
        elif 'net' in name:
            model.geom_rgba[index] = [.90, .94, .94, .08]
            changes.append([name, 'translucent net finish, unchanged collision'])
    return changes


def append_visual_box(scene, mujoco, pos, size, rgba):
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError('Insufficient visual decoration capacity')
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX,
                       np.asarray(size, dtype=float), np.asarray(pos, dtype=float),
                       np.eye(3).reshape(-1), np.asarray(rgba, dtype=float))
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    scene.ngeom += 1


def decorate_scene(scene, mujoco):
    # These objects are render-only and never enter the native collision model.
    for x in (-1.03, 1.03):
        for y in (-.57, .57):
            append_visual_box(scene, mujoco, [x, y, .37], [.035, .035, .37], [.13, .19, .21, 1.])
    for y in (-.752, .752):
        append_visual_box(scene, mujoco, [0., y, .7603], [1.36, .006, .0003], [.93, .97, .96, 1.])
    for x in (-1.36, 1.36):
        append_visual_box(scene, mujoco, [x, 0., .7603], [.006, .752, .0003], [.93, .97, .96, 1.])
    append_visual_box(scene, mujoco, [0., 0., .7603], [1.36, .004, .0003], [.93, .97, .96, 1.])
    # Fine visual mesh occupies the existing net's bounds; the original physical
    # collider stays present and unchanged, with a translucent finish.
    for y in np.linspace(-.915, .915, 47):
        append_visual_box(scene, mujoco, [0., y, .83625], [.0006, .0006, .07625], [.80, .88, .89, .55])
    for z in np.linspace(.76, .9125, 7):
        append_visual_box(scene, mujoco, [0., 0., z], [.0006, .915, .0006], [.80, .88, .89, .55])
    append_visual_box(scene, mujoco, [0., 0., .9125], [.004, .915, .003], [.94, .97, .97, 1.])


def make_plan(arrays, shot: str, seconds: float, fps: int, source_start: float, speed: float):
    trials = np.unique(arrays['trial_id']).astype(int).tolist()
    # Hard cuts separate independent serves. No state is stitched across a feed reset.
    segment = 2. if shot == 'wide' else seconds
    plan = []
    for frame in range(round(seconds * fps)):
        movie_time = frame / fps
        slot = int(movie_time / segment) if shot == 'wide' else 0
        trial = trials[slot % len(trials)]
        timestamp = source_start + (movie_time % segment) * speed
        index = recorded_frame_index(arrays, trial, timestamp)
        plan.append({'video_frame': frame, 'video_time': movie_time,
                     'trial_id': trial, 'recorded_index': index,
                     'trial_time': float(arrays['trial_time'][index]),
                     'cut_before': frame == 0 or (shot == 'wide' and frame % round(segment * fps) == 0)})
    return plan


def render(args):
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import mujoco
    from PIL import Image

    model_path, recording_path = Path(args.model), Path(args.recording)
    model = load_visual_model(model_path, mujoco)
    for index in range(model.nbody):
        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index) or '').lower()
        if 'wall' in name or 'backboard' in name:
            raise ValueError('No-wall footage requires a model compiled without the wall body')
    with np.load(recording_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    required = {'qpos': model.nq, 'qvel': model.nv, 'ctrl': model.nu}
    count = len(arrays['trial_time'])
    for key, width in required.items():
        if arrays[key].shape != (count, width) or not np.isfinite(arrays[key]).all():
            raise ValueError('Invalid recorded state array: ' + key)
    appearance = decorate_model(model, mujoco)
    data = mujoco.MjData(model)
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    plan = make_plan(arrays, args.shot, args.seconds, args.fps, args.source_start, args.speed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    process = None
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    try:
        if not args.preview:
            ffmpeg = find_ffmpeg(args.ffmpeg)
            process = subprocess.Popen([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
                '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{args.width}x{args.height}',
                '-r', str(args.fps), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast',
                '-crf', '18', '-threads', '4', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                windows_destination(output, ffmpeg)], stdin=subprocess.PIPE)
        for frame, item in enumerate(plan):
            index = item['recorded_index']
            data.qpos[:] = arrays['qpos'][index]
            data.qvel[:] = arrays['qvel'][index]
            data.ctrl[:] = arrays['ctrl'][index]
            data.time = float(arrays['trial_time'][index])
            mujoco.mj_forward(model, data)
            camera = camera_for(mujoco, args.shot, frame / max(len(plan) - 1, 1))
            renderer.update_scene(data, camera=camera, scene_option=options)
            decorate_scene(renderer.scene, mujoco)
            image = renderer.render()
            if args.preview:
                Image.fromarray(image).save(output)
                plan = plan[:1]
                break
            process.stdin.write(image.tobytes())
            if frame % args.fps == 0:
                print(json.dumps({'rendered_frames': frame, 'total_frames': len(plan), 'shot': args.shot}), flush=True)
        if process:
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError('ffmpeg encoding failed')
    finally:
        renderer.close()
        if process and process.poll() is None:
            process.terminate()
    receipt = {'renderer_sha256': sha256(Path(__file__)), 'mujoco_version': mujoco.__version__,
               'model_sha256': sha256(model_path), 'recording_sha256': sha256(recording_path),
               'output_sha256': sha256(output), 'resolution': [args.width, args.height],
               'fps': args.fps, 'shot': args.shot, 'playback_speed': args.speed,
               'state_interpolation': False, 'physics_steps_during_render': 0,
               'appearance_only': appearance + [['decoration', 'table legs, painted boundary lines, fine net mesh, studio skybox and diagonal studio key light']],
               'frames': plan}
    output.with_suffix('.render.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'frames': len(plan), 'sha256': receipt['output_sha256']}), flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--model', default='assets/scene.xml')
    result.add_argument('--recording', default='outputs/demo/session.npz')
    result.add_argument('--output', required=True)
    result.add_argument('--shot', choices=['wide', 'orbit', 'close'], default='wide')
    result.add_argument('--seconds', type=float, default=16.)
    result.add_argument('--fps', type=int, default=30)
    result.add_argument('--width', type=int, default=1920)
    result.add_argument('--height', type=int, default=1080)
    result.add_argument('--source-start', type=float, default=.02)
    result.add_argument('--speed', type=float, default=1.)
    result.add_argument('--preview', action='store_true')
    result.add_argument('--ffmpeg')
    return result


if __name__ == '__main__':
    render(parser().parse_args())
