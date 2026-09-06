"""Calculate FVD between generated and ground-truth video directories."""

import argparse
import os

import torch
import torchvision.io as io
import torchvision.transforms as transforms
from tqdm import tqdm


def to_bcthw(videos):
    """Convert a BTCHW tensor to BCTHW."""
    if videos.shape[-3] == 1:
        videos = videos.repeat(1, 1, 3, 1, 1)
    return videos.permute(0, 2, 1, 3, 4)


def load_videos_from_folder(
    folder,
    resize=(64, 64),
    max_videos=None,
    max_frames=None,
    device="cpu",
):
    """Load sorted MP4 files into a padded BTCHW tensor in [0, 1]."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Video directory not found: {folder}")

    video_files = sorted(
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.lower().endswith(".mp4")
    )
    if max_videos is not None:
        video_files = video_files[:max_videos]
    if not video_files:
        raise ValueError(f"No MP4 videos found in: {folder}")

    resize_transform = transforms.Resize(resize)
    videos = []
    for video_file in tqdm(video_files, desc=f"Loading {folder}"):
        try:
            video, _, _ = io.read_video(video_file, pts_unit="sec")
            video = video.float().permute(0, 3, 1, 2) / 255.0
            video = resize_transform(video)
            if max_frames is not None:
                video = video[:max_frames]
            if len(video) == 0:
                raise ValueError("decoded video has no frames")
            videos.append(video)
        except Exception as error:
            print(f"[WARN] Failed to read {video_file}: {error}")

    if not videos:
        raise ValueError(f"No videos could be decoded from: {folder}")

    max_length = max(video.shape[0] for video in videos)
    padded_videos = []
    for video in videos:
        padding_length = max_length - video.shape[0]
        if padding_length > 0:
            padding = torch.zeros(
                (padding_length, *video.shape[1:]), dtype=video.dtype
            )
            video = torch.cat([video, padding], dim=0)
        padded_videos.append(video)

    print(f"Loaded {len(padded_videos)} videos from {folder}")
    return torch.stack(padded_videos, dim=0).to(device)


def calculate_fvd(videos1, videos2, device, method):
    if videos1.shape != videos2.shape:
        raise ValueError(
            "Generated and GT video tensors must have the same shape; "
            f"got {tuple(videos1.shape)} and {tuple(videos2.shape)}"
        )

    if method == "styleganv":
        from fvd.styleganv.fvd import (
            frechet_distance,
            get_fvd_feats,
            load_i3d_pretrained,
        )
    elif method == "videogpt":
        from fvd.videogpt.fvd import (
            frechet_distance,
            get_fvd_logits,
            load_i3d_pretrained,
        )

        get_fvd_feats = get_fvd_logits
    else:
        raise ValueError(f"Unsupported FVD method: {method}")

    print(f"Calculating FVD using {method}...")
    i3d = load_i3d_pretrained(device=device)
    features1 = get_fvd_feats(to_bcthw(videos1).cpu(), i3d=i3d, device=device)
    features2 = get_fvd_feats(to_bcthw(videos2).cpu(), i3d=i3d, device=device)
    return frechet_distance(features1, features2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate FVD between generated and GT video folders."
    )
    parser.add_argument(
        "--generated-videos",
        "--folder1",
        dest="generated_videos",
        required=True,
        help="Directory containing generated MP4 videos.",
    )
    parser.add_argument(
        "--gt-videos",
        "--folder2",
        dest="gt_videos",
        required=True,
        help="Directory containing ground-truth MP4 videos.",
    )
    parser.add_argument(
        "--num-videos",
        "--data_num",
        dest="num_videos",
        type=int,
        default=None,
        help="Use at most the first N sorted videos (default: all).",
    )
    parser.add_argument(
        "--max-frames",
        "--max_frames",
        dest="max_frames",
        type=int,
        default=40,
        help="Use at most this many frames per video (default: 40).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("videogpt", "styleganv"),
        default=("videogpt", "styleganv"),
        help="FVD implementations to run (default: both).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda when available, otherwise cpu).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_videos is not None and args.num_videos <= 0:
        raise ValueError("--num-videos must be a positive integer")
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be a positive integer")

    videos1 = load_videos_from_folder(
        args.generated_videos,
        max_videos=args.num_videos,
        max_frames=args.max_frames,
        device=args.device,
    )
    videos2 = load_videos_from_folder(
        args.gt_videos,
        max_videos=args.num_videos,
        max_frames=args.max_frames,
        device=args.device,
    )

    for method in args.methods:
        value = calculate_fvd(videos1, videos2, args.device, method)
        print(f"[FVD - {method}] = {value}")


if __name__ == "__main__":
    main()
