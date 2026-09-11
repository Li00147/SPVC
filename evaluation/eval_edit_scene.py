"""Run batched SPVC inference on the PanopticFix edit-scene benchmark."""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SPVC_ROOT = Path(__file__).resolve().parents[1]
spvc_root_str = str(SPVC_ROOT)
if spvc_root_str in sys.path:
    sys.path.remove(spvc_root_str)
sys.path.insert(0, spvc_root_str)

import torch
from PIL import Image

from diffsynth.core.loader.config import ModelConfig
from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.utils.data import VideoData, save_video

DEFAULT_BENCHMARK_ROOT = SPVC_ROOT / "SPVC-PanopticFix-Benchmark"
DEFAULT_PROMPT = (
    "Transform a low-quality autonomous driving video into a high-quality, "
    "realistic driving video with clear details and consistent motion."
)
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

DATASET_RESOLUTIONS: Dict[str, Tuple[int, int]] = {
    "nuscenes": (800, 464),
    "waymo": (960, 640),
    "pandaset": (960, 544),
}
DATASET_OUTPUT_RESOLUTIONS: Dict[str, Tuple[int, int]] = {
    "nuscenes": (800, 450),
    "waymo": (960, 640),
    "pandaset": (960, 540),
}


def resize_frame(frame: Image.Image, size_wh: Tuple[int, int]) -> Image.Image:
    if frame.size == size_wh:
        return frame.convert("RGB")
    return frame.convert("RGB").resize(size_wh, Image.Resampling.LANCZOS)


def parse_frame_from_name(stem: str) -> int:
    """Read the leading frame index from names such as 016_000.png."""
    frame = stem.split("_", maxsplit=1)[0]
    if not frame.isdigit():
        raise ValueError(
            f"Invalid frame name {stem!r}; expected a leading numeric frame index"
        )
    return int(frame)


def collect_image_frames(
    clip_dir: Path, infer_size_wh: Tuple[int, int]
) -> List[Image.Image]:
    indexed_paths = []
    for path in clip_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            frame_index = parse_frame_from_name(path.stem)
        except ValueError:
            continue
        indexed_paths.append((frame_index, path))

    if not indexed_paths:
        raise RuntimeError(
            f"No valid image frames found in {clip_dir}; expected names such as "
            "000_000.png"
        )

    indexed_paths.sort(key=lambda item: (item[0], str(item[1])))
    seen_indices = set()
    frames = []
    for frame_index, path in indexed_paths:
        if frame_index in seen_indices:
            raise RuntimeError(
                f"Multiple images use frame index {frame_index} in {clip_dir}. "
                "Each clip must contain one camera stream."
            )
        seen_indices.add(frame_index)
        with Image.open(path) as image:
            frames.append(resize_frame(image, infer_size_wh))
    return frames


def collect_video_frames(
    video_path: Path, infer_size_wh: Tuple[int, int]
) -> List[Image.Image]:
    video = VideoData(str(video_path))
    frame_count = len(video)
    if frame_count <= 0:
        raise RuntimeError(f"Video has no decodable frames: {video_path}")
    return [resize_frame(video[index], infer_size_wh) for index in range(frame_count)]


def load_clip_frames(
    clip_input: Path, infer_size_wh: Tuple[int, int]
) -> List[Image.Image]:
    if clip_input.is_dir():
        return collect_image_frames(clip_input, infer_size_wh)
    return collect_video_frames(clip_input, infer_size_wh)


def clip_name(clip_input: Path) -> str:
    return clip_input.name if clip_input.is_dir() else clip_input.stem


def list_clip_inputs(input_root: Path) -> List[Path]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_root}")

    candidates = sorted(
        (
            path
            for path in input_root.iterdir()
            if path.is_dir()
            or (path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)
        ),
        key=lambda path: clip_name(path),
    )
    if not candidates:
        raise RuntimeError(
            f"No video files or clip image directories found in: {input_root}"
        )

    names = [clip_name(path) for path in candidates]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise RuntimeError(f"Duplicate clip names in {input_root}: {duplicates}")
    return candidates


def parse_scenes(value: str) -> Optional[set[str]]:
    value = value.strip()
    if value.lower() == "all":
        return None
    scenes = {
        token.zfill(3) if token.isdigit() else token
        for token in (part.strip() for part in value.split(","))
        if token
    }
    if not scenes:
        raise ValueError("--scenes must be 'all' or a comma-separated scene list")
    return scenes


def matches_scene(name: str, scenes: Optional[set[str]]) -> bool:
    return scenes is None or any(name.startswith(f"{scene}_") for scene in scenes)


def scene_id_from_clip(name: str) -> str:
    scene_id = name.split("_", maxsplit=1)[0]
    if not scene_id.isdigit():
        raise ValueError(
            f"Cannot parse scene ID from clip name {name!r}; expected <scene>_<id>"
        )
    return scene_id


def find_reference_video(gt_videos_dir: Path, scene_id: str) -> Optional[Path]:
    if not gt_videos_dir.is_dir():
        return None
    candidates = sorted(gt_videos_dir.glob(f"{scene_id}_*.mp4"))
    if candidates:
        return candidates[0]
    exact = gt_videos_dir / f"{scene_id}.mp4"
    return exact if exact.is_file() else None


def load_reference_frames(
    video_path: Path, infer_size_wh: Tuple[int, int], num_frames: int
) -> List[Image.Image]:
    frames = collect_video_frames(video_path, infer_size_wh)
    if len(frames) < num_frames:
        frames.extend([frames[-1]] * (num_frames - len(frames)))
    return frames[:num_frames]


def sliding_windows(n_frames: int, window: int, overlap: int) -> List[Tuple[int, int]]:
    if n_frames <= 0:
        return []
    if n_frames <= window:
        return [(0, n_frames)]

    stride = window - overlap
    starts = list(range(0, n_frames - window + 1, stride))
    final_start = n_frames - window
    if starts[-1] != final_start:
        starts.append(final_start)
    return [(start, start + window) for start in starts]


def build_pipeline(args, device: str) -> WanVideoPipeline:
    vram_config = {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cpu",
        "onload_dtype": torch.bfloat16,
        "onload_device": device,
        "preparing_dtype": torch.bfloat16,
        "preparing_device": device,
        "computation_dtype": torch.bfloat16,
        "computation_device": device,
    }
    model_patterns = (
        "high_noise_model/diffusion_pytorch_model*.safetensors",
        "low_noise_model/diffusion_pytorch_model*.safetensors",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth",
    )
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(
                model_id=args.model_id,
                origin_file_pattern=pattern,
                **vram_config,
            )
            for pattern in model_patterns
        ],
        vram_limit=args.vram_limit,
    )
    pipe.load_lora(pipe.dit, str(args.ckpt_high_noise), alpha=args.lora_alpha)
    pipe.load_lora(pipe.dit2, str(args.ckpt_low_noise), alpha=args.lora_alpha)
    return pipe


def run_sliding_inference(
    pipe: WanVideoPipeline,
    name: str,
    input_frames: List[Image.Image],
    reference_frames: Optional[List[Image.Image]],
    videos_dir: Path,
    images_dir: Path,
    output_size_wh: Tuple[int, int],
    args,
) -> Path:
    windows = sliding_windows(len(input_frames), args.win, args.overlap)
    print(
        f"[Clip] {name}: frames={len(input_frames)}, windows={windows}, "
        f"reference={'yes' if reference_frames is not None else 'no'}"
    )

    stitched_frames = []
    covered_until = 0
    for start, end in windows:
        window_frames = input_frames[start:end]
        if len(window_frames) < args.win:
            window_frames = window_frames + [window_frames[-1]] * (
                args.win - len(window_frames)
            )

        with torch.inference_mode():
            generated = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                control_video=window_frames,
                reference_video=reference_frames,
                height=DATASET_RESOLUTIONS[args.dataset_name][1],
                width=DATASET_RESOLUTIONS[args.dataset_name][0],
                num_frames=args.win,
                seed=args.seed,
                cfg_scale=args.cfg_scale,
                switch_DiT_boundary=args.switch_dit_boundary,
                num_inference_steps=args.num_inference_steps,
                sigma_shift=args.sigma_shift,
                tiled=args.tiled,
                tile_size=tuple(args.tile_size),
                tile_stride=tuple(args.tile_stride),
            )
        if len(generated) < args.win:
            raise RuntimeError(
                f"Pipeline returned {len(generated)} frames for a {args.win}-frame window"
            )

        generated = [resize_frame(frame, output_size_wh) for frame in generated]
        overlap_with_output = max(0, covered_until - start)
        new_frame_count = end - max(start, covered_until)
        stitched_frames.extend(
            generated[
                overlap_with_output : overlap_with_output + new_frame_count
            ]
        )
        covered_until = max(covered_until, end)

    output_video = videos_dir / f"{name}.mp4"
    save_video(stitched_frames, str(output_video), fps=args.fps, quality=args.quality)
    for index, frame in enumerate(stitched_frames):
        frame.save(images_dir / f"{name}_frame_{index:04d}.png")
    return output_video


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run sliding-window SPVC inference on PanopticFix MP4 files or legacy "
            "clip image directories."
        )
    )
    parser.add_argument(
        "--dataset-name",
        "--dataset_name",
        dest="dataset_name",
        choices=sorted(DATASET_RESOLUTIONS),
        default="nuscenes",
    )
    parser.add_argument(
        "--input-dir",
        "--input_dir",
        dest="input_dir",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT / "testset",
        help="Directory containing MP4 files or per-clip image directories.",
    )
    parser.add_argument(
        "--output-root",
        "--output_root",
        dest="output_root",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT / "results",
        help="Output root; generated files are written under images/ and videos/.",
    )
    parser.add_argument(
        "--gt-videos-dir",
        "--gt_videos_dir",
        dest="gt_videos_dir",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT / "gt" / "videos",
    )
    parser.add_argument(
        "--ckpt-high-noise",
        "--ckpt_high_noise",
        dest="ckpt_high_noise",
        type=Path,
        default=SPVC_ROOT / "checkpoints" / "spvc_high_noise.safetensors",
   
    )
    parser.add_argument(
        "--ckpt-low-noise",
        "--ckpt_low_noise",
        dest="ckpt_low_noise",
        type=Path,
        default=SPVC_ROOT / "checkpoints" / "spvc_low_noise.safetensors",

    )
    parser.add_argument(
        "--scenes",
        default="all",
        help="Scene IDs: all, 062, or a comma-separated list such as 002,062.",
    )
    parser.add_argument(
        "--disable-reference",
        "--disable_reference",
        dest="disable_reference",
        action="store_true",
        help="Run without the clean reference-video condition.",
    )
    parser.add_argument("--model-id", default="PAI/Wan2.2-Fun-A14B-Control")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--vram-limit", type=float)
    parser.add_argument("--win", type=int, default=25)
    parser.add_argument("--overlap", type=int, default=10)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--sigma-shift", type=float, default=5.0)
    parser.add_argument("--switch-dit-boundary", type=float, default=0.875)
    parser.add_argument("--lora-alpha", type=float, default=1.0)
    parser.add_argument(
        "--tiled", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--tile-size", type=int, nargs=2, default=(30, 52))
    parser.add_argument("--tile-stride", type=int, nargs=2, default=(15, 26))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.win <= 0 or (args.win - 1) % 4 != 0:
        raise ValueError("--win must be positive and satisfy (win - 1) % 4 == 0")
    if args.overlap < 0 or args.overlap >= args.win:
        raise ValueError("--overlap must satisfy 0 <= overlap < win")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")

    for checkpoint in (args.ckpt_high_noise, args.ckpt_low_noise):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"LoRA checkpoint not found: {checkpoint}")
    if not args.disable_reference and not args.gt_videos_dir.is_dir():
        raise FileNotFoundError(
            f"GT video directory not found: {args.gt_videos_dir}"
        )

    scenes = parse_scenes(args.scenes)
    clip_inputs = [
        path
        for path in list_clip_inputs(args.input_dir)
        if matches_scene(clip_name(path), scenes)
    ]
    if not clip_inputs:
        raise RuntimeError("No clips matched --input-dir and --scenes")

    videos_dir = args.output_root / "videos"
    images_dir = args.output_root / "images"
    videos_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_root}")
    print(f"Dataset:          {args.dataset_name}")
    print(f"Device:           {device}")
    print(f"Matched clips:    {len(clip_inputs)}")
    if not args.disable_reference:
        print("Reference policy: first sorted GT video for each scene")

    pipe = build_pipeline(args, device)
    infer_size_wh = DATASET_RESOLUTIONS[args.dataset_name]
    output_size_wh = DATASET_OUTPUT_RESOLUTIONS[args.dataset_name]
    reference_cache: Dict[str, Optional[List[Image.Image]]] = {}

    for clip_input in clip_inputs:
        name = clip_name(clip_input)
        input_frames = load_clip_frames(clip_input, infer_size_wh)
        reference_frames = None
        if not args.disable_reference:
            scene_id = scene_id_from_clip(name)
            if scene_id not in reference_cache:
                reference_path = find_reference_video(args.gt_videos_dir, scene_id)
                if reference_path is None:
                    print(
                        f"[WARN] No reference video for scene {scene_id}; "
                        "running that scene without reference"
                    )
                    reference_cache[scene_id] = None
                else:
                    print(f"[Reference] scene {scene_id}: {reference_path.name}")
                    reference_cache[scene_id] = load_reference_frames(
                        reference_path, infer_size_wh, args.win
                    )
            reference_frames = reference_cache[scene_id]

        output = run_sliding_inference(
            pipe=pipe,
            name=name,
            input_frames=input_frames,
            reference_frames=reference_frames,
            videos_dir=videos_dir,
            images_dir=images_dir,
            output_size_wh=output_size_wh,
            args=args,
        )
        print(f"[Saved] {output}")

    print("PanopticFix inference finished.")


if __name__ == "__main__":
    main()
