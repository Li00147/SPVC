import argparse
import random
import sys
from pathlib import Path


MODE_DIRS = {
    "cross-reference": "cross_reference_data",
    "underfitting": "underfitting_data",
    "random-mask": "random_mask_data",
}


def parse_list(values):
    if not values:
        return None
    result = []
    for value in values:
        result.extend(part.strip() for part in value.split(",") if part.strip())
    return list(dict.fromkeys(result))


def find_checkpoints(root, checkpoint_name, scenes):
    candidates = sorted(root.iterdir()) if scenes is None else [root / scene for scene in scenes]
    checkpoints = []
    for scene_dir in candidates:
        path = scene_dir / checkpoint_name
        if path.is_file():
            checkpoints.append(path)
        else:
            print(f"Skipping missing checkpoint: {path}")
    return checkpoints


def mask_images(image_dir, seed):
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in extensions):
        with Image.open(path) as source:
            image = source.convert("RGB")
        width, height = image.size
        draw = ImageDraw.Draw(image)
        color_mode = rng.choice(("black", "white", "random"))
        color = {"black": (0, 0, 0), "white": (255, 255, 255)}.get(
            color_mode, tuple(rng.randint(0, 255) for _ in range(3))
        )
        shape = rng.choice(("rectangle", "ellipse", "polygon"))
        if shape == "rectangle":
            box_width = rng.randint(max(1, width // 10), max(1, width // 3))
            box_height = rng.randint(max(1, height // 10), max(1, height // 3))
            left = rng.randint(0, max(0, width - box_width))
            top = rng.randint(0, max(0, height - box_height))
            draw.rectangle((left, top, left + box_width, top + box_height), fill=color)
        elif shape == "ellipse":
            radius = rng.randint(max(1, min(width, height) // 10), max(1, min(width, height) // 4))
            center_x = rng.randint(radius, max(radius, width - radius))
            center_y = rng.randint(radius, max(radius, height - radius))
            draw.ellipse(
                (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
                fill=color,
            )
        else:
            points = [(rng.randint(0, width - 1), rng.randint(0, height - 1)) for _ in range(rng.randint(3, 8))]
            draw.polygon(points, fill=color)
        save_args = {"quality": 95} if path.suffix.lower() in {".jpg", ".jpeg"} else {}
        image.save(path, **save_args)


def load_drivestudio(root):
    sys.path.insert(0, str(root.resolve()))
    from datasets.driving_dataset import DrivingDataset
    from models.video_utils import render_images, save_videos
    from omegaconf import OmegaConf
    from utils.misc import import_str

    return DrivingDataset, render_images, save_videos, OmegaConf, import_str


def render(args):
    import torch

    root = Path(args.drivestudio_root)
    if not (root / "datasets" / "driving_dataset.py").is_file():
        raise FileNotFoundError(f"Invalid DriveStudio root: {root}")
    DrivingDataset, render_images, save_videos, OmegaConf, import_str = load_drivestudio(root)
    scenes = parse_list(args.scenes)
    cameras = [int(value) for value in parse_list(args.cameras) or []]
    if args.mode == "cross-reference" and not cameras:
        cameras = [0, 3, 4]
    checkpoint_name = args.checkpoint
    if checkpoint_name is None:
        checkpoint_name = "checkpoint_01000.pth" if args.mode == "underfitting" else "checkpoint_final.pth"
    checkpoints = find_checkpoints(Path(args.checkpoint_root), checkpoint_name, scenes)
    if not checkpoints:
        raise FileNotFoundError(f"No {checkpoint_name} files found under {args.checkpoint_root}")
    output_dir = Path(args.output_root) / MODE_DIRS[args.mode] / args.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    for checkpoint in checkpoints:
        scene = checkpoint.parent.name
        config_path = checkpoint.parent / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        print(f"Rendering {args.mode}: dataset={args.dataset_name}, scene={scene}")
        config = OmegaConf.load(config_path)
        if cameras:
            config.data.pixel_source.cameras = cameras
        dataset = DrivingDataset(data_cfg=config.data)
        trainer = import_str(config.trainer.type)(
            **config.trainer,
            num_timesteps=dataset.num_img_timesteps,
            model_config=config.model,
            num_train_images=len(dataset.train_image_set),
            num_full_images=len(dataset.full_image_set),
            test_set_indices=dataset.test_timesteps,
            scene_aabb=dataset.get_aabb().reshape(2, 3),
            device=device,
        )
        trainer.resume_from_checkpoint(ckpt_path=str(checkpoint), load_only_model=True)
        trainer.set_eval()
        with torch.no_grad():
            results = render_images(
                trainer=trainer,
                dataset=dataset.full_image_set,
                compute_metrics=False,
                compute_error_map=bool(OmegaConf.select(config, "render.vis_error", default=False)),
                compute_fid=False,
            )
            save_videos(
                results,
                str(output_dir / f"{scene}.mp4"),
                layout=dataset.layout,
                num_timestamps=dataset.num_img_timesteps,
                keys=["rgbs"],
                save_images=True,
                num_cams=dataset.pixel_source.num_cams,
                save_seperate_video=bool(OmegaConf.select(config, "logging.save_seperate_video", default=False)),
                fps=int(OmegaConf.select(config, "render.fps", default=args.fps)),
                verbose=True,
            )
        del results, trainer, dataset
        if args.mode == "random-mask":
            image_dir = output_dir / f"{scene}_rgbs"
            if not image_dir.is_dir():
                raise FileNotFoundError(f"DriveStudio did not create {image_dir}")
            mask_images(image_dir, args.seed + int(scene) if scene.isdigit() else args.seed)


def main():
    parser = argparse.ArgumentParser(description="Render SPVC degradation images with DriveStudio checkpoints")
    parser.add_argument("--mode", choices=sorted(MODE_DIRS), required=True)
    parser.add_argument("--drivestudio-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-name", choices=("nuscenes", "pandaset", "waymo"), required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--scenes", nargs="*")
    parser.add_argument("--cameras", nargs="*")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    render(parser.parse_args())


if __name__ == "__main__":
    main()
