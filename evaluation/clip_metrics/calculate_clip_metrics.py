"""Compute scene/group-averaged CLIP-F and CLIP-V from image directories."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute CLIP-F and CLIP-V from generated and GT images."
    )
    parser.add_argument(
        "--generated-images",
        type=Path,
        required=True,
        help="Directory containing generated images.",
    )
    parser.add_argument(
        "--gt-images",
        type=Path,
        required=True,
        help="Directory containing ground-truth images.",
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
        help="Group by the filename prefix before '_' or use one global group.",
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
        raise RuntimeError(f"No supported image files found in: {folder}")

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
        with torch.inference_mode():
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
    return (features[:-1] * features[1:]).sum(-1).mean().item()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")

    try:
        import clip
    except ImportError as error:
        raise RuntimeError(
            "OpenAI CLIP is not installed. Run: "
            "python -m pip install git+https://github.com/openai/CLIP.git"
        ) from error

    generated_groups = load_images_by_group(args.generated_images, args.grouping)
    gt_groups = load_images_by_group(args.gt_images, args.grouping)

    print(f"Generated images: {args.generated_images}")
    print(f"GT images:        {args.gt_images}")
    print(f"Loading CLIP model {args.clip_model!r} on {args.device}")
    model, preprocess = clip.load(args.clip_model, device=args.device)
    model.eval()

    group_clip_f = []
    group_clip_v = []
    matched_images = 0
    for group_id in sorted(generated_groups):
        if group_id not in gt_groups:
            print(f"[WARN] Skipping group {group_id}: no GT images")
            continue

        generated_images = generated_groups[group_id]
        gt_images = gt_groups[group_id]
        if len(generated_images) < 2 or len(gt_images) < 2:
            print(f"[WARN] Skipping group {group_id}: fewer than two images")
            continue
        if len(generated_images) != len(gt_images):
            print(
                f"[WARN] Group {group_id} has {len(generated_images)} generated "
                f"and {len(gt_images)} GT images; comparing the first "
                f"{min(len(generated_images), len(gt_images))} sorted pairs"
            )

        generated_features = encode(
            generated_images, model, preprocess, args.batch_size, args.device
        )
        gt_features = encode(
            gt_images, model, preprocess, args.batch_size, args.device
        )
        group_clip_f.append(clip_f(generated_features, gt_features))
        group_clip_v.append(clip_v(generated_features))
        matched_images += min(len(generated_images), len(gt_images))

    if not group_clip_f:
        raise RuntimeError(
            "No matching generated/GT groups were found. Check paths, names, "
            "and --grouping."
        )

    print(f"Matched groups:  {len(group_clip_f)}")
    print(f"Matched images:  {matched_images}")
    print(f"[CLIP-F] = {np.mean(group_clip_f):.4f}")
    print(f"[CLIP-V] = {np.mean(group_clip_v):.4f}")


if __name__ == "__main__":
    main()
