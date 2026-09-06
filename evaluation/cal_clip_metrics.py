"""Compute scene/group-averaged CLIP-F and CLIP-V for SPVC results."""

import argparse
from collections import defaultdict
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_benchmark_root = script_dir.parent / "SPVC-Benchamrk"

    parser = argparse.ArgumentParser(
        description="Compute CLIP-F and CLIP-V for an SPVC benchmark dataset."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset folder name, for example nuscenes, waymo, or pandaset.",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=default_benchmark_root,
        help="Benchmark root containing dataset folders.",
    )
    parser.add_argument(
        "--gen-images-root",
        type=Path,
        default=None,
        help="Override <benchmark>/<dataset>/results/images.",
    )
    parser.add_argument(
        "--gt-images-root",
        type=Path,
        default=None,
        help="Override <benchmark>/<dataset>/gt/gt_images.",
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        default=["4.0"],
        help="Scale subfolders to evaluate (default: 4.0).",
    )
    parser.add_argument(
        "--clip-model",
        default="ViT-B/32",
        help="OpenAI CLIP model name or local checkpoint path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="CLIP image encoding batch size (default: 64).",
    )
    parser.add_argument(
        "--grouping",
        choices=("prefix", "all"),
        default="prefix",
        help="Average by filename prefix before '_' or treat all images as one group.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda when available, otherwise cpu).",
    )
    return parser.parse_args()


def get_group_id(path, grouping):
    if grouping == "all":
        return "all"
    return path.name.split("_", maxsplit=1)[0]


def load_images_by_group(folder, grouping):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Image directory not found: {folder}")

    paths = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise RuntimeError(f"No image files found in: {folder}")

    groups = defaultdict(list)
    for path in paths:
        groups[get_group_id(path, grouping)].append(path)
    return groups


def preprocess_image(path, preprocess):
    with Image.open(path) as image:
        return preprocess(image.convert("RGB")).unsqueeze(0)


def encode(paths, model, preprocess, batch_size, device):
    features = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        images = torch.cat(
            [preprocess_image(path, preprocess) for path in batch_paths], dim=0
        ).to(device)
        with torch.no_grad():
            batch_features = model.encode_image(images)
            batch_features = batch_features / batch_features.norm(
                dim=-1, keepdim=True
            )
        features.append(batch_features.cpu())
    return torch.cat(features, dim=0)


def clip_f(generated_features, gt_features):
    count = min(len(generated_features), len(gt_features))
    return (
        (generated_features[:count] * gt_features[:count])
        .sum(-1)
        .mean()
        .item()
    )


def clip_v(features):
    if len(features) < 2:
        return np.nan
    return (features[:-1] * features[1:]).sum(-1).mean().item()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer")
    if not args.dataset or any(char in args.dataset for char in "/\\"):
        raise ValueError("--dataset must be a dataset folder name, not a path")

    dataset_root = args.benchmark_root / args.dataset
    gen_images_root = args.gen_images_root or dataset_root / "results" / "images"
    gt_images_root = args.gt_images_root or dataset_root / "gt" / "gt_images"

    print(f"Dataset: {args.dataset}")
    print(f"Generated images root: {gen_images_root}")
    print(f"GT images root: {gt_images_root}")
    print(f"Loading CLIP model {args.clip_model!r} on {args.device}")
    model, preprocess = clip.load(args.clip_model, device=args.device)
    gt_groups = load_images_by_group(gt_images_root, args.grouping)

    for scale in args.scales:
        gen_folder = gen_images_root / scale
        gen_groups = load_images_by_group(gen_folder, args.grouping)
        group_clip_f = []
        group_clip_v = []

        print(f"\n========== Scale {scale} ==========")
        for group_id in sorted(gen_groups):
            if group_id not in gt_groups:
                print(f"[WARN] Skipping group {group_id}: no GT images")
                continue

            gen_images = gen_groups[group_id]
            gt_images = gt_groups[group_id]
            if len(gen_images) < 2 or len(gt_images) < 2:
                print(f"[WARN] Skipping group {group_id}: fewer than two images")
                continue

            gen_features = encode(
                gen_images, model, preprocess, args.batch_size, args.device
            )
            gt_features = encode(
                gt_images, model, preprocess, args.batch_size, args.device
            )
            group_clip_f.append(clip_f(gen_features, gt_features))
            group_clip_v.append(clip_v(gen_features))

        if not group_clip_f:
            raise RuntimeError(
                "No matching generated/GT groups were found. Check paths, names, "
                "and --grouping."
            )

        print(f"[{scale}] CLIP-F (group avg): {np.mean(group_clip_f):.4f}")
        print(f"[{scale}] CLIP-V (group avg): {np.mean(group_clip_v):.4f}")


if __name__ == "__main__":
    main()
