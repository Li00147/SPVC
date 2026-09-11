"""Render one nuScenes map-projection diagnostic image."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes


CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one nuScenes HD-map projection on a camera image."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--version", default="advanced_12Hz_trainval")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--camera", choices=CAMERAS, default="CAM_FRONT")
    parser.add_argument(
        "--layers",
        nargs="+",
        default=("road_segment", "ped_crossing"),
    )
    parser.add_argument(
        "--map-name",
        help="Optional map name; defaults to the location recorded for the sample.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nusc = NuScenes(
        version=args.version,
        dataroot=str(args.data_root),
        verbose=False,
    )
    if not 0 <= args.sample_index < len(nusc.sample):
        raise ValueError(
            f"--sample-index must be in [0, {len(nusc.sample)}), "
            f"got {args.sample_index}"
        )

    sample = nusc.sample[args.sample_index]
    if args.map_name is None:
        scene = nusc.get("scene", sample["scene_token"])
        log = nusc.get("log", scene["log_token"])
        map_name = log["location"]
    else:
        map_name = args.map_name

    args.output.parent.mkdir(parents=True, exist_ok=True)
    nusc_map = NuScenesMap(dataroot=str(args.data_root), map_name=map_name)
    nusc_map.render_map_in_image(
        nusc,
        sample["token"],
        layer_names=list(args.layers),
        camera_channel=args.camera,
        out_path=str(args.output),
        verbose=False,
    )
    print(f"Saved map projection to {args.output}")


if __name__ == "__main__":
    main()
