"""Render one nuScenes vehicle-box diagnostic image."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.nuscenes import NuScenes


CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
)
EXCLUDED_CATEGORY_PREFIXES = (
    "animal",
    "human.",
    "movable_object.",
    "static_object.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render vehicle 3D boxes on the first frame of a scene."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--version", default="advanced_12Hz_trainval")
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--camera", choices=CAMERAS, default="CAM_FRONT")
    parser.add_argument("--line-width", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.line_width <= 0:
        raise ValueError("--line-width must be positive")

    nusc = NuScenes(
        version=args.version,
        dataroot=str(args.data_root),
        verbose=False,
    )
    if not 0 <= args.scene_index < len(nusc.scene):
        raise ValueError(
            f"--scene-index must be in [0, {len(nusc.scene)}), "
            f"got {args.scene_index}"
        )

    scene = nusc.scene[args.scene_index]
    sample = nusc.get("sample", scene["first_sample_token"])
    camera_data = nusc.get("sample_data", sample["data"][args.camera])

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    nusc.render_sample_data(camera_data["token"], with_anns=False, ax=ax)
    _, boxes, camera_intrinsic = nusc.get_sample_data(camera_data["token"])
    for box in boxes:
        if box.name.startswith(EXCLUDED_CATEGORY_PREFIXES):
            continue
        color = np.asarray(nusc.explorer.get_color(box.name)) / 255.0
        box.render(
            ax,
            view=camera_intrinsic,
            normalize=True,
            colors=(color, color, color),
            linewidth=args.line_width,
        )

    ax.set_title("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved vehicle-box projection to {args.output}")


if __name__ == "__main__":
    main()
