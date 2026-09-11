import argparse
import os
import re
import shutil
import subprocess
import time
from typing import Dict, List

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes


VERSION = 'advanced_12Hz_trainval'
FRAMES_PER_VIDEO = 25
DEFAULT_FPS = 12
MAP_LAYERS = ['road_segment', 'ped_crossing']

# Camera indices follow the order used by nuScenes sample['data'].
CAMERAS = [
    'CAM_FRONT',          # 0
    'CAM_FRONT_RIGHT',    # 1
    'CAM_BACK_RIGHT',     # 2
    'CAM_BACK',           # 3
    'CAM_BACK_LEFT',      # 4
    'CAM_FRONT_LEFT',     # 5
]

# Keep the filtering behavior from the previous combined renderer: vehicles only.
EXCLUDED_CATEGORY_PREFIXES = (
    'animal',
    'human.',
    'movable_object.',
    'static_object.',
)


def scene_output_id(scene: dict) -> int:
    match = re.search(r'(\d+)$', scene['name'])
    if match is None:
        raise ValueError(
            f"Scene name must end with a numeric ID, got {scene['name']!r}"
        )
    return int(match.group(1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Render nuScenes map overlays and vehicle boxes to 25-frame MP4 clips.',
    )
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--version', default=VERSION)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--start-sequence', type=int, default=0, help='Inclusive, zero-based scene index.')
    parser.add_argument('--end-sequence', type=int, default=None, help='Exclusive, zero-based scene index.')
    parser.add_argument('--fps', type=int, default=DEFAULT_FPS)
    parser.add_argument(
        '--ffmpeg',
        default=shutil.which('ffmpeg'),
        help='ffmpeg executable; defaults to the executable discovered from PATH.',
    )
    parser.add_argument(
        '--camera-indices',
        type=int,
        nargs='+',
        choices=range(len(CAMERAS)),
        default=list(range(len(CAMERAS))),
        help='Camera indices to render; defaults to all six cameras.',
    )
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='List planned outputs without rendering.')
    parser.add_argument(
        '--max-clips-per-sequence',
        type=int,
        default=None,
        help='Optional limit for testing; defaults to every complete clip.',
    )
    return parser.parse_args()


def scene_sample_tokens(nusc: NuScenes, scene: dict) -> List[str]:
    tokens = []
    token = scene['first_sample_token']
    while token:
        tokens.append(token)
        token = nusc.get('sample', token)['next']
    return tokens


def remove_camera_background(ax) -> None:
    """Remove the raster camera image while keeping projected map artists."""
    for image_artist in list(ax.get_images()):
        image_artist.remove()


def render_frame(
    nusc: NuScenes,
    nusc_map: NuScenesMap,
    sample_token: str,
    camera_channel: str,
) -> np.ndarray:
    sample = nusc.get('sample', sample_token)
    camera_token = sample['data'][camera_channel]

    fig, ax = nusc_map.render_map_in_image(
        nusc,
        sample_token,
        layer_names=MAP_LAYERS,
        camera_channel=camera_channel,
        verbose=False,
    )
    remove_camera_background(ax)

    _, boxes, camera_intrinsic = nusc.get_sample_data(camera_token)
    for box in boxes:
        if box.name.startswith(EXCLUDED_CATEGORY_PREFIXES):
            continue
        color = np.array(nusc.explorer.get_color(box.name)) / 255.0
        box.render(
            ax,
            view=camera_intrinsic,
            normalize=True,
            colors=(color, color, color),
            linewidth=1,
        )

    ax.set_title('')
    ax.axis('off')
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


def write_clip(
    nusc: NuScenes,
    nusc_map: NuScenesMap,
    sample_tokens: List[str],
    camera_channel: str,
    output_path: str,
    fps: int,
    ffmpeg: str,
) -> None:
    temporary_path = output_path + '.part.mp4'
    process = None

    try:
        for sample_token in sample_tokens:
            frame = render_frame(nusc, nusc_map, sample_token, camera_channel)
            height, width = frame.shape[:2]

            if process is None:
                command = [
                    ffmpeg, '-loglevel', 'error', '-y',
                    '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                    '-s:v', f'{width}x{height}', '-r', str(fps), '-i', '-',
                    '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                    temporary_path,
                ]
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

            process.stdin.write(frame.tobytes())

        process.stdin.close()
        stderr = process.stderr.read().decode('utf-8', errors='replace')
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f'ffmpeg exited with code {return_code}: {stderr}')
        os.replace(temporary_path, output_path)
    except Exception:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError('--fps must be positive')
    if not args.dry_run and args.ffmpeg is None:
        raise FileNotFoundError(
            'ffmpeg was not found in PATH; install it or pass --ffmpeg /path/to/ffmpeg'
        )
    if args.max_clips_per_sequence is not None and args.max_clips_per_sequence <= 0:
        raise ValueError('--max-clips-per-sequence must be positive')

    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)
    end_sequence = len(nusc.scene) if args.end_sequence is None else args.end_sequence
    if not 0 <= args.start_sequence <= end_sequence <= len(nusc.scene):
        raise ValueError(
            f'Expected 0 <= start <= end <= {len(nusc.scene)}, got '
            f'{args.start_sequence}, {end_sequence}',
        )

    os.makedirs(args.output_dir, exist_ok=True)
    map_cache: Dict[str, NuScenesMap] = {}
    planned = 0
    completed = 0
    skipped = 0
    start_time = time.time()

    for sequence_idx in range(args.start_sequence, end_sequence):
        scene = nusc.scene[sequence_idx]
        scene_id = scene_output_id(scene)
        sample_tokens = scene_sample_tokens(nusc, scene)
        usable_frame_count = len(sample_tokens) // FRAMES_PER_VIDEO * FRAMES_PER_VIDEO
        dropped_frame_count = len(sample_tokens) - usable_frame_count

        log = nusc.get('log', scene['log_token'])
        location = log['location']
        if location not in map_cache and not args.dry_run:
            map_cache[location] = NuScenesMap(dataroot=args.data_root, map_name=location)

        print(
            f'sequence {sequence_idx}/{end_sequence - 1}: {scene["name"]}, '
            f'{len(sample_tokens)} frames, dropping {dropped_frame_count}',
            flush=True,
        )

        for clip_idx, start_frame_idx in enumerate(range(0, usable_frame_count, FRAMES_PER_VIDEO)):
            if args.max_clips_per_sequence is not None and clip_idx >= args.max_clips_per_sequence:
                break
            clip_tokens = sample_tokens[start_frame_idx:start_frame_idx + FRAMES_PER_VIDEO]
            for camera_idx in args.camera_indices:
                camera_channel = CAMERAS[camera_idx]
                planned += 1
                filename = (
                    f'combined_{scene_id:04d}_{start_frame_idx:03d}_'
                    f'{camera_idx}.mp4'
                )
                output_path = os.path.join(args.output_dir, filename)

                if args.dry_run:
                    print(output_path)
                    continue
                if os.path.exists(output_path) and not args.overwrite:
                    skipped += 1
                    continue

                print(f'  rendering {filename} ({camera_channel})', flush=True)
                write_clip(
                    nusc,
                    map_cache[location],
                    clip_tokens,
                    camera_channel,
                    output_path,
                    args.fps,
                    args.ffmpeg,
                )
                completed += 1

    elapsed = time.time() - start_time
    print(
        f'done: planned={planned}, completed={completed}, skipped={skipped}, '
        f'elapsed_seconds={elapsed:.1f}',
        flush=True,
    )


if __name__ == '__main__':
    main()
