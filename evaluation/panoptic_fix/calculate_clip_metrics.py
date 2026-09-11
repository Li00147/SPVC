"""Compute PanopticFix CLIP-F and CLIP-V with edit-aware grouping."""

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute PanopticFix CLIP-F and CLIP-V. Generated frames are "
            "grouped by (scene, edit variant), while GT frames are grouped by scene."
        )
    )
    parser.add_argument(
        "--generated-images",
        "--gen_dir",
        dest="generated_images",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--gt-images",
        "--gt_dir",
        dest="gt_images",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--clip-model",
        "--model",
        dest="clip_model",
        default="ViT-B/32",
        help="OpenAI CLIP model name or local checkpoint path.",
    )
    parser.add_argument("--batch-size", "--batch_size", type=int, default=64)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every scene/edit score.",
    )
    return parser.parse_args()


def natural_key(path: Path):
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", path.name)
    ]


def list_images(folder: Path) -> List[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Image directory not found: {folder}")
    paths = sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=natural_key,
    )
    if not paths:
        raise RuntimeError(f"No supported images found in: {folder}")
    return paths


def gt_scene_id(path: Path) -> str:
    scene_id = path.stem.split("_", maxsplit=1)[0]
    if not scene_id:
        raise ValueError(f"Cannot parse scene ID from GT image: {path.name}")
    return scene_id


def generated_scene_variant(path: Path) -> Tuple[str, str]:
    parts = path.stem.split("_")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Cannot parse scene/edit IDs from generated image: {path.name}. "
            "Expected names such as 002_0_frame_0000.png."
        )
    return parts[0], parts[1]


def load_gt_by_scene(folder: Path) -> Dict[str, List[Path]]:
    groups: DefaultDict[str, List[Path]] = defaultdict(list)
    for path in list_images(folder):
        groups[gt_scene_id(path)].append(path)
    for paths in groups.values():
        paths.sort(key=natural_key)
    return dict(groups)


def load_generated_by_variant(
    folder: Path,
) -> Dict[Tuple[str, str], List[Path]]:
    groups: DefaultDict[Tuple[str, str], List[Path]] = defaultdict(list)
    for path in list_images(folder):
        groups[generated_scene_variant(path)].append(path)
    for paths in groups.values():
        paths.sort(key=natural_key)
    return dict(groups)


def preprocess_image(path: Path, preprocess) -> torch.Tensor:
    with Image.open(path) as image:
        return preprocess(image.convert("RGB")).unsqueeze(0)


@torch.inference_mode()
def encode(
    paths: List[Path],
    model,
    preprocess,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    features = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        images = torch.cat(
            [preprocess_image(path, preprocess) for path in batch_paths], dim=0
        ).to(device)
        batch_features = model.encode_image(images)
        batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
        features.append(batch_features.cpu())
    return torch.cat(features, dim=0)


def clip_f(generated_features: torch.Tensor, gt_features: torch.Tensor) -> float:
    count = min(len(generated_features), len(gt_features))
    if count == 0:
        return float("nan")
    return (
        (generated_features[:count] * gt_features[:count])
        .sum(dim=-1)
        .mean()
        .item()
    )


def clip_v(features: torch.Tensor) -> float:
    if len(features) < 2:
        return float("nan")
    return (features[:-1] * features[1:]).sum(dim=-1).mean().item()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")

    try:
        import clip
    except ImportError as error:
        raise RuntimeError(
            "OpenAI CLIP is not installed. Install evaluation/requirements.txt."
        ) from error

    gt_by_scene = load_gt_by_scene(args.gt_images)
    generated_by_variant = load_generated_by_variant(args.generated_images)
    variants_by_scene: DefaultDict[str, List[str]] = defaultdict(list)
    for scene_id, variant_id in generated_by_variant:
        variants_by_scene[scene_id].append(variant_id)
    for scene_id in variants_by_scene:
        variants_by_scene[scene_id] = sorted(
            set(variants_by_scene[scene_id]),
            key=lambda value: (
                (0, int(value)) if value.isdigit() else (1, value.lower())
            ),
        )

    common_scenes = sorted(set(variants_by_scene) & set(gt_by_scene))
    if not common_scenes:
        raise RuntimeError(
            "No matching generated/GT scenes. Generated names should look like "
            "002_0_frame_0000.png and GT names should begin with 002_."
        )

    print(f"Generated images: {args.generated_images}")
    print(f"GT images:        {args.gt_images}")
    print(f"Loading CLIP model {args.clip_model!r} on {args.device}")
    model, preprocess = clip.load(args.clip_model, device=args.device)
    model.eval()

    scene_clip_f = []
    scene_clip_v = []
    matched_variants = 0
    matched_frames = 0
    for scene_id in common_scenes:
        gt_paths = gt_by_scene[scene_id]
        gt_features = encode(
            gt_paths, model, preprocess, args.batch_size, args.device
        )
        variant_clip_f = []
        variant_clip_v = []
        for variant_id in variants_by_scene[scene_id]:
            generated_paths = generated_by_variant[(scene_id, variant_id)]
            generated_features = encode(
                generated_paths, model, preprocess, args.batch_size, args.device
            )
            score_f = clip_f(generated_features, gt_features)
            score_v = clip_v(generated_features)
            if not np.isnan(score_f):
                variant_clip_f.append(score_f)
            if not np.isnan(score_v):
                variant_clip_v.append(score_v)
            matched_variants += 1
            matched_frames += min(len(generated_paths), len(gt_paths))
            if len(generated_paths) != len(gt_paths):
                print(
                    f"[WARN] Scene {scene_id}, edit {variant_id}: "
                    f"generated={len(generated_paths)}, GT={len(gt_paths)}; "
                    f"CLIP-F uses {min(len(generated_paths), len(gt_paths))} frames"
                )
            if args.verbose:
                print(
                    f"[Scene {scene_id} | Edit {variant_id}] "
                    f"CLIP-F={score_f:.4f}, CLIP-V={score_v:.4f}"
                )

        if variant_clip_f:
            scene_clip_f.append(float(np.mean(variant_clip_f)))
        if variant_clip_v:
            scene_clip_v.append(float(np.mean(variant_clip_v)))
        if args.verbose:
            print(
                f"[Scene {scene_id}] edits={len(variants_by_scene[scene_id])}, "
                f"CLIP-F={np.mean(variant_clip_f):.4f}, "
                f"CLIP-V={np.mean(variant_clip_v):.4f}"
            )

    if not scene_clip_f or not scene_clip_v:
        raise RuntimeError("No valid scene scores were produced")

    print(f"Matched scenes:   {len(common_scenes)}")
    print(f"Matched edits:    {matched_variants}")
    print(f"Matched frames:   {matched_frames}")
    print(f"[CLIP-F] scene average = {np.mean(scene_clip_f):.4f}")
    print(f"[CLIP-V] scene average = {np.mean(scene_clip_v):.4f}")


if __name__ == "__main__":
    main()
