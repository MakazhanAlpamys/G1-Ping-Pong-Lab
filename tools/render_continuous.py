"""One uninterrupted camera over an actual continuous recorded simulation.

Global physical time advances at 1x. Each rendered frame uses an exact saved
state, with no interpolation, montage edits, simulation steps or hidden resets.
New-ball feed events belong to the source simulation and its physical report.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np

HELPERS = Path(__file__).with_name('render_footage.py')
_spec = importlib.util.spec_from_file_location('g1_footage_visual_helpers', HELPERS)
visual = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(visual)


def global_frame_index(times, timestamp: float) -> int:
    times = np.asarray(times)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Continuous recording requires strictly increasing global physical time')
    return int(np.clip(np.searchsorted(times, timestamp, side='right') - 1, 0, len(times) - 1))


def continuous_plan(arrays, seconds: float | None, fps: int):
    times = np.asarray(arrays['time'])
    global_frame_index(times, float(times[0]))
    step = float(np.median(np.diff(times))) if len(times) > 1 else 1. / fps
    available = float(times[-1] - times[0] + step)
    seconds = available if seconds is None else seconds
    if fps <= 0 or seconds <= 0 or seconds > available + 1.e-6:
        raise ValueError('Requested duration must fit the complete continuous recording')
    result = []
    for frame in range(round(seconds * fps)):
        requested = float(times[0]) + frame / fps
        index = global_frame_index(times, requested)
        result.append({'video_frame': frame, 'video_time': frame / fps,
                       'recorded_index': index, 'physical_time': float(times[index]),
                       'cut_before': frame == 0})
    return result


def continuous_camera(mujoco, progress: float):
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [-.50, 0., .67]
    camera.distance = 4.25
    camera.azimuth = 138. - 38. * progress
    camera.elevation = -19. + 2. * progress
    return camera


def validate_continuity_report(report):
    for key in ['robot_resets_after_start', 'wbc_history_resets_after_start']:
        if key not in report or report[key] != 0:
            raise ValueError('Continuous footage requires an explicit zero reset count: ' + key)


def validate_input_provenance(report, model_path, recording_path):
    for key, path in [('scene_sha256', model_path), ('trajectory_sha256', recording_path)]:
        if key not in report or report[key] != visual.sha256(Path(path)):
            raise ValueError('The continuity report does not match the physical input: ' + key)


def render(args):
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import mujoco
    from PIL import Image

    model_path, recording_path = Path(args.model), Path(args.recording)
    source_report = json.loads(Path(args.report).read_text())
    validate_continuity_report(source_report)
    validate_input_provenance(source_report, model_path, recording_path)
    model = visual.load_visual_model(model_path, mujoco)
    for index in range(model.nbody):
        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index) or '').lower()
        if 'wall' in name or 'backboard' in name:
            raise ValueError('The physical wall must be absent from a no-wall recording')
    with np.load(recording_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    count = len(arrays['time'])
    for key, width in {'qpos': model.nq, 'qvel': model.nv, 'ctrl': model.nu}.items():
        if arrays[key].shape != (count, width) or not np.isfinite(arrays[key]).all():
            raise ValueError('Invalid native recorded array: ' + key)
    plan = continuous_plan(arrays, args.seconds, args.fps)
    appearance = visual.decorate_model(model, mujoco)
    data = mujoco.MjData(model)
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    process = None
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    try:
        if not args.preview:
            ffmpeg = visual.find_ffmpeg(args.ffmpeg)
            process = subprocess.Popen([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
                '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{args.width}x{args.height}',
                '-r', str(args.fps), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast',
                '-crf', '18', '-threads', '4', '-g', str(args.fps), '-keyint_min', str(args.fps),
                '-sc_threshold', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                visual.windows_destination(output, ffmpeg)], stdin=subprocess.PIPE)
        for frame, item in enumerate(plan):
            index = item['recorded_index']
            data.qpos[:] = arrays['qpos'][index]
            data.qvel[:] = arrays['qvel'][index]
            data.ctrl[:] = arrays['ctrl'][index]
            data.time = item['physical_time']
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=continuous_camera(mujoco, frame / max(len(plan) - 1, 1)),
                                  scene_option=options)
            visual.decorate_scene(renderer.scene, mujoco)
            image = renderer.render()
            if args.preview:
                Image.fromarray(image).save(output)
                plan = plan[:1]
                break
            process.stdin.write(image.tobytes())
            if frame % args.fps == 0:
                print(json.dumps({'rendered_frames': frame, 'total_frames': len(plan),
                                  'physical_time': item['physical_time']}), flush=True)
        if process:
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError('Continuous recording encoding failed')
    finally:
        renderer.close()
        if process and process.poll() is None:
            process.terminate()
    receipt = {'renderer_sha256': visual.sha256(Path(__file__)),
               'visual_helpers_sha256': visual.sha256(HELPERS),
               'mujoco_version': mujoco.__version__, 'model_sha256': visual.sha256(model_path),
               'recording_sha256': visual.sha256(recording_path), 'output_sha256': visual.sha256(output),
               'resolution': [args.width, args.height], 'fps': args.fps, 'shot': 'continuous',
               'playback_speed': 1., 'state_interpolation': False, 'physics_steps_during_render': 0,
               'montage_cuts': 0, 'camera_changes': 0,
               'appearance_only': appearance + [['decoration', 'table legs, painted lines, net mesh, skybox and key light']],
               'frames': plan}
    receipt['source_report_sha256'] = visual.sha256(Path(args.report))
    receipt['robot_resets_after_start'] = source_report['robot_resets_after_start']
    receipt['wbc_history_resets_after_start'] = source_report['wbc_history_resets_after_start']
    receipt['ball_replacements_after_start_in_source'] = source_report.get('ball_replacements_after_start')
    receipt['source_feed_launch_times'] = [feed['launched_at'] for feed in source_report.get('feeds', [])]
    output.with_suffix('.render.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'frames': len(plan), 'sha256': receipt['output_sha256']}), flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--model', default='assets/scene.xml')
    result.add_argument('--recording', default='outputs/continuous/session.npz')
    result.add_argument('--report', required=True)
    result.add_argument('--output', required=True)
    result.add_argument('--shot', choices=['continuous'], default='continuous')
    result.add_argument('--seconds', type=float)
    result.add_argument('--fps', type=int, default=30)
    result.add_argument('--width', type=int, default=1920)
    result.add_argument('--height', type=int, default=1080)
    result.add_argument('--preview', action='store_true')
    result.add_argument('--ffmpeg')
    return result


if __name__ == '__main__':
    render(parser().parse_args())
