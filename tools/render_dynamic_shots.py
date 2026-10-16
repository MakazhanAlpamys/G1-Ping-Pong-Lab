"""Short, exact-state camera shots for an explicitly edited controlled-feed demo.

Independent feeds are separate shots. This utility does not simulate physics,
invent ball positions, interpolate states or imply a continuous same-ball rally.
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
_spec = importlib.util.spec_from_file_location('dynamic_visual_helpers', HELPERS)
visual = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(visual)

# azimuth, elevation, distance, fovy, world look-at, small camera arc.
CAMERAS = {
    'landscape': [
        (138., -20., 4.35, 40., [-.4, 0., .67], -5.),
        (120., -12., 2.15, 40., [-1.35, -.20, .95], 4.),
        (178., -13., 4.65, 40., [-.20, 0., .75], -3.),
        (137., -42., 4.85, 40., [-.20, 0., .62], -4.),
        (155., -15., 4.05, 43., [-.75, -.05, .83], 4.),
        (207., -10., 2.4, 43., [-1.35, -.20, .95], 4.),
        (187., -17., 4.90, 38., [-.10, -.12, .73], 3.),
        (112., -29., 4.95, 40., [-.20, 0., .66], -5.),
    ],
    'portrait': [
        (165., -22., 5.5, 48., [-.3, 0., .65], -4.),
        (155., -10., 2.6, 55., [-1.35, -.20, 1.0], 4.),
        (180., -13., 4.9, 48., [-.2, 0., .68], -3.),
        (169., -43., 5.9, 48., [-.2, 0., .58], -4.),
        (172., -17., 4.7, 50., [-.55, -.08, .75], 4.),
        (205., -9., 2.8, 55., [-1.35, -.20, 1.0], 4.),
        (187., -15., 5.1, 47., [-.1, -.1, .67], 3.),
        (151., -32., 6.0, 51., [-.2, 0., .62], -4.),
    ],
}


def dynamic_plan(arrays, *, trial: int, start: float, seconds: float, fps: int, speed: float):
    ids = np.flatnonzero(np.asarray(arrays['trial_id']) == trial)
    if not len(ids) or fps <= 0 or speed <= 0 or seconds <= 0:
        raise ValueError('A real recorded trial and positive timing parameters are required')
    times = np.asarray(arrays['trial_time'])[ids]
    if start < float(times[0]) - 1.e-8 or start + seconds * speed > float(times[-1]) + 1.e-8:
        raise ValueError('The entire shot must fit within one recorded physical feed')
    result = []
    for frame in range(round(seconds * fps)):
        requested = start + frame / fps * speed
        index = visual.recorded_frame_index(arrays, trial, requested)
        result.append({'video_frame': frame, 'video_time': frame / fps,
                       'recorded_index': index, 'trial_id': trial,
                       'trial_time': float(arrays['trial_time'][index])})
    return result


def camera_parameters(format_name, shot, replay=False):
    azimuth, elevation, distance, fovy, lookat, arc = CAMERAS[format_name][shot]
    if replay:
        azimuth, elevation, distance, fovy, lookat, arc = (
            174., -15., 5.45 if format_name == 'portrait' else 5.05,
            40. if format_name == 'portrait' else 33., [-.35, -.10, .77], -6.)
    return azimuth, elevation, distance, fovy, lookat, arc


def camera_for(mujoco, model, format_name, shot, progress, replay=False):
    azimuth, elevation, distance, fovy, lookat, arc = camera_parameters(format_name, shot, replay)
    model.vis.global_.fovy = fovy
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth + arc * progress
    camera.elevation = elevation
    return camera


def orange_ball_bounds(image):
    values = image.astype(np.float32)
    r, g, b = values[..., 0], values[..., 1], values[..., 2]
    mask = (r > 150) & (g > 35) & (g < 190) & (b < 95) & (r > 1.25 * g) & (g > 1.35 * b)
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return None
    return {'pixels': int(len(xx)), 'bounds': [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]}


def load_inputs(args, mujoco):
    model_path, recording_path = Path(args.model), Path(args.recording)
    model = visual.load_visual_model(model_path, mujoco)
    for index in range(model.nbody):
        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index) or '').lower()
        if 'wall' in name or 'backboard' in name:
            raise ValueError('No-wall shots require the actual wall to be absent')
    with np.load(recording_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    count = len(arrays['trial_time'])
    for key, width in {'qpos': model.nq, 'qvel': model.nv, 'ctrl': model.nu}.items():
        if arrays[key].shape != (count, width) or not np.isfinite(arrays[key]).all():
            raise ValueError('Invalid recorded native state: ' + key)
    for trial in range(8):
        visual.recorded_frame_index(arrays, trial, .5)
    appearance = visual.decorate_model(model, mujoco)
    return model, arrays, appearance


def draw_frame(renderer, model, data, arrays, index, format_name, shot, progress, replay, mujoco, options):
    data.qpos[:] = arrays['qpos'][index]
    data.qvel[:] = arrays['qvel'][index]
    data.ctrl[:] = arrays['ctrl'][index]
    data.time = float(arrays['trial_time'][index])
    mujoco.mj_forward(model, data)
    camera = camera_for(mujoco, model, format_name, shot, progress, replay)
    renderer.update_scene(data, camera=camera, scene_option=options)
    visual.decorate_scene(renderer.scene, mujoco)
    return renderer.render()


def render_format(args, format_name, model, arrays, appearance, mujoco):
    from PIL import Image, ImageDraw
    width, height = (1920, 1080) if format_name == 'landscape' else (1080, 1920)
    model.vis.global_.offwidth = max(width, 1920)
    model.vis.global_.offheight = max(height, 1920)
    data = mujoco.MjData(model)
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    renderer = mujoco.Renderer(model, height=height, width=width)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    preview_items = []
    preview_images = []
    try:
        if args.preview:
            preview_dir = output_root / 'previews'
            preview_dir.mkdir(parents=True, exist_ok=True)
            for shot in range(8):
                for timestamp in ([.48, .65] if shot in (1, 5) else [.48, .90]):
                    index = visual.recorded_frame_index(arrays, shot, timestamp)
                    image = draw_frame(renderer, model, data, arrays, index, format_name, shot,
                                       (timestamp - .10) / 1.1, False, mujoco, options)
                    path = preview_dir / f'{format_name}-shot-{shot + 1:02d}-{timestamp:.2f}.png'
                    Image.fromarray(image).save(path)
                    preview_items.append({'shot': shot + 1, 'trial_id': shot,
                                          'trial_time': float(arrays['trial_time'][index]),
                                          'orange_ball': orange_ball_bounds(image), 'file': path.name})
                    thumb = Image.fromarray(image)
                    thumb.thumbnail((480, 270) if format_name == 'landscape' else (270, 480))
                    draw = ImageDraw.Draw(thumb)
                    draw.rectangle((0, 0, 110, 20), fill='#10171e')
                    draw.text((5, 4), f'{shot + 1:02d} / {timestamp:.2f}s', fill='white')
                    preview_images.append(thumb)
            cell_width, cell_height = (480, 270) if format_name == 'landscape' else (270, 480)
            sheet = Image.new('RGB', (cell_width * 4, cell_height * 4), '#10171e')
            for index, image in enumerate(preview_images):
                sheet.paste(image, ((index % 4) * cell_width, (index // 4) * cell_height))
            sheet.save(preview_dir / f'{format_name}-contact.png')
            (preview_dir / f'{format_name}-preview.json').write_text(json.dumps(preview_items, indent=2) + '\n')
            print(json.dumps({'preview': format_name, 'frames': len(preview_items)}), flush=True)
            return
        target = output_root / format_name
        target.mkdir(parents=True, exist_ok=True)
        shots = [(shot, False) for shot in range(8)] + [(4, True)]
        ffmpeg = visual.find_ffmpeg(args.ffmpeg)
        for shot, replay in shots:
            name = 'replay' if replay else f'shot-{shot + 1:02d}'
            start, seconds, speed = (.20, 2.4, .5) if replay else (.10, 1.1, 1.)
            plan = dynamic_plan(arrays, trial=shot, start=start, seconds=seconds, fps=args.fps, speed=speed)
            output = target / (name + '.mp4')
            process = subprocess.Popen([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
                '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
                '-r', str(args.fps), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast',
                '-crf', '18', '-threads', '4', '-g', str(args.fps), '-keyint_min', str(args.fps),
                '-sc_threshold', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                visual.windows_destination(output, ffmpeg)], stdin=subprocess.PIPE)
            try:
                for frame, item in enumerate(plan):
                    image = draw_frame(renderer, model, data, arrays, item['recorded_index'], format_name,
                                       shot, frame / max(len(plan) - 1, 1), replay, mujoco, options)
                    process.stdin.write(image.tobytes())
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError('Shot encoding failed: ' + name)
            finally:
                if process.poll() is None:
                    process.terminate()
            receipt = {'renderer_sha256': visual.sha256(Path(__file__)),
                       'visual_helpers_sha256': visual.sha256(HELPERS),
                       'model_sha256': visual.sha256(Path(args.model)),
                       'recording_sha256': visual.sha256(Path(args.recording)),
                       'output_sha256': visual.sha256(output), 'mujoco_version': mujoco.__version__,
                       'format': format_name, 'resolution': [width, height], 'fps': args.fps,
                       'trial_id': shot, 'source_start': start, 'playback_speed': speed,
                       'video_seconds': seconds, 'independent_controlled_feed': True,
                       'replay': replay, 'state_interpolation': False,
                       'physics_steps_during_render': 0, 'appearance_only': appearance,
                       'camera_spec': camera_parameters(format_name, shot, replay), 'frames': plan}
            output.with_suffix('.render.json').write_text(json.dumps(receipt, indent=2) + '\n')
            print(json.dumps({'format': format_name, 'shot': name, 'frames': len(plan),
                              'sha256': receipt['output_sha256']}), flush=True)
    finally:
        renderer.close()


def main(args):
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import mujoco
    model, arrays, appearance = load_inputs(args, mujoco)
    formats = ['landscape', 'portrait'] if args.format == 'both' else [args.format]
    for format_name in formats:
        render_format(args, format_name, model, arrays, appearance, mujoco)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--model', default='assets/scene.xml')
    result.add_argument('--recording', default='data/demo-session.npz')
    result.add_argument('--output-root', default='outputs/dynamic')
    result.add_argument('--format', choices=['landscape', 'portrait', 'both'], default='both')
    result.add_argument('--fps', type=int, default=30)
    result.add_argument('--preview', action='store_true')
    result.add_argument('--ffmpeg')
    return result


if __name__ == '__main__':
    main(parser().parse_args())
