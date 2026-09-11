"""Run SPVC novel-view inference on a benchmark directory."""

import argparse
import re
import sys
from dataclasses import dataclass
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
from diffsynth.models.camera_pose_encoder import load_camera_pose_encoder
from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.utils.data import VideoData, save_video


DEFAULT_BENCHMARK_ROOT = SPVC_ROOT / "SPVC-Benchamrk" / "nuscenes"
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
DEFAULT_INFER_SIZE = (800, 464)
DEFAULT_OUTPUT_SIZE = (800, 450)
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
CONTROL_NAME_PATTERN = re.compile(
    r"^(?P<scene>\d+)_(?P<camera>[^_]+)_(?P<direction>[^_]+)_"
    r"(?P<scale>\d+(?:\.\d+)?)_(?P<clip>\d+)_?$"
)


@dataclass(frozen=True)
class BenchmarkClip:
    input_path: Path
    scene_id: str
    camera_id: str
    direction: str
    clip_id: str
    scale: str

    @property
    def name(self) -> str:
        return self.input_path.stem

    @property
    def start_frame(self) -> int:
        return int(self.clip_id) * 25


@dataclass(frozen=True)
class ConditionPaths:
    control_video: Path
    reference_video: Path
    structured_condition: Path
    camera_pose: Path


def parse_control_video(path: Path) -> BenchmarkClip:
    match = CONTROL_NAME_PATTERN.fullmatch(path.stem)
    if match is None:
        raise ValueError(
            f"Invalid control-video name: {path.name}. Expected "
            "<scene>_<camera>_<direction>_<scale>_<clip>_.mp4"
        )
    return BenchmarkClip(
        input_path=path,
        scene_id=match.group("scene"),
        camera_id=match.group("camera"),
        direction=match.group("direction"),
        clip_id=match.group("clip"),
        scale=match.group("scale"),
    )


def parse_csv_filter(value: str, pad_numeric_to: Optional[int] = None) -> Optional[set[str]]:
    if value.strip().lower() == "all":
        return None
    values = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if pad_numeric_to is not None and item.isdigit():
            item = item.zfill(pad_numeric_to)
        values.add(item)
    if not values:
        raise ValueError("A filter must be 'all' or a comma-separated list")
    return values


def discover_clips(
    input_dir: Path,
    scenes: Optional[set[str]],
    scales: Optional[set[str]],
) -> List[BenchmarkClip]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    clips = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        clip = parse_control_video(path)
        if scenes is not None and clip.scene_id not in scenes:
            continue
        if scales is not None and clip.scale not in scales:
            continue
        clips.append(clip)

    if not clips:
        raise RuntimeError("No videos matched --input-dir, --scenes, and --scales")
    return clips


def reference_video_path(gt_videos_dir: Path, clip: BenchmarkClip) -> Path:
    return gt_videos_dir / f"{clip.scene_id}_{clip.clip_id}.mp4"


def structured_condition_path(
    structured_condition_dir: Path, clip: BenchmarkClip
) -> Path:
    return structured_condition_dir / (
        f"combined_{int(clip.scene_id):04d}_{clip.start_frame:03d}_0_"
        f"{clip.direction}_{clip.scale}.mp4"
    )


def camera_pose_path(camera_pose_dir: Path, clip: BenchmarkClip) -> Path:
    return camera_pose_dir / f"{clip.name}.pt"


def condition_paths(
    clip: BenchmarkClip,
    gt_videos_dir: Path,
    structured_condition_dir: Path,
    camera_pose_dir: Path,
) -> ConditionPaths:
    return ConditionPaths(
        control_video=clip.input_path,
        reference_video=reference_video_path(gt_videos_dir, clip),
        structured_condition=structured_condition_path(
            structured_condition_dir, clip
        ),
        camera_pose=camera_pose_path(camera_pose_dir, clip),
    )


def load_camera_pose(path: Path, num_frames: int) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Camera pose not found: {path}")
    pose = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(pose, dict):
        for key in ("cam_pose", "rel_pose", "pose"):
            if key in pose:
                pose = pose[key]
                break
    if not isinstance(pose, torch.Tensor):
        raise TypeError(f"Camera pose must be a tensor: {path}")
    if pose.ndim not in (3, 4) or pose.shape[-2:] != (4, 4):
        raise ValueError(
            f"Camera pose must have shape (F,4,4) or (B,F,4,4), "
            f"got {tuple(pose.shape)}: {path}"
        )
    if pose.shape[-3] != num_frames:
        raise ValueError(
            f"Camera pose has {pose.shape[-3]} frames, but {num_frames} are "
            f"required: {path}"
        )
    if not bool(torch.isfinite(pose).all()):
        raise ValueError(f"Camera pose contains NaN or Inf: {path}")
    return pose.to(dtype=torch.float32)


def validate_and_load_conditions(
    clips: List[BenchmarkClip],
    gt_videos_dir: Path,
    structured_condition_dir: Path,
    camera_pose_dir: Path,
    num_frames: int,
) -> Dict[str, Tuple[ConditionPaths, torch.Tensor]]:
    resolved = {
        clip.name: condition_paths(
            clip,
            gt_videos_dir,
            structured_condition_dir,
            camera_pose_dir,
        )
        for clip in clips
    }
    missing = []
    for paths in resolved.values():
        for label, path in (
            ("control video", paths.control_video),
            ("reference video", paths.reference_video),
            ("structured condition", paths.structured_condition),
            ("camera pose", paths.camera_pose),
        ):
            if not path.is_file():
                missing.append(f"{label}: {path}")
    if missing:
        preview = "\n".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
        raise FileNotFoundError(
            f"Missing {len(missing)} required condition files:\n{preview}{suffix}"
        )

    return {
        name: (paths, load_camera_pose(paths.camera_pose, num_frames))
        for name, paths in resolved.items()
    }


def print_conditions(
    index: int,
    total: int,
    paths: ConditionPaths,
) -> None:
    print(f"[{index}/{total}] Conditions")
    print(f"  control_video:       {paths.control_video}")
    print(f"  reference_video:     {paths.reference_video}")
    print(f"  structured_condition: {paths.structured_condition}")
    print(f"  camera_pose:         {paths.camera_pose}")


def resize_frame(frame: Image.Image, size_wh: Tuple[int, int]) -> Image.Image:
    if frame.size == size_wh:
        return frame.convert("RGB")
    return frame.convert("RGB").resize(size_wh, Image.Resampling.LANCZOS)


def load_video_frames(
    path: Path,
    size_wh: Tuple[int, int],
    num_frames: int,
) -> List[Image.Image]:
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    video = VideoData(str(path))
    if len(video) < num_frames:
        raise RuntimeError(
            f"{path} has {len(video)} frames, but {num_frames} are required"
        )
    return [resize_frame(video[index], size_wh) for index in range(num_frames)]


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


def build_condition_kwargs(
    paths: ConditionPaths,
    camera_pose: torch.Tensor,
    infer_size: Tuple[int, int],
    num_frames: int,
) -> Dict[str, object]:
    return {
        "control_video": load_video_frames(
            paths.control_video, infer_size, num_frames
        ),
        "reference_video": load_video_frames(
            paths.reference_video, infer_size, num_frames
        ),
        "reference_combined_video": load_video_frames(
            paths.structured_condition, infer_size, num_frames
        ),
        "cam_pose": camera_pose,
    }


def save_outputs(
    frames: List[Image.Image],
    clip: BenchmarkClip,
    output_root: Path,
    output_size: Tuple[int, int],
    fps: int,
    quality: int,
) -> Path:
    videos_dir = output_root / "videos" / clip.scale
    images_dir = output_root / "images" / clip.scale
    videos_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    resized_frames = [resize_frame(frame, output_size) for frame in frames]
    output_video = videos_dir / f"{clip.name}.mp4"
    save_video(resized_frames, str(output_video), fps=fps, quality=quality)
    for index, frame in enumerate(resized_frames):
        frame.save(images_dir / f"{clip.name}_frame_{index:04d}.png")
    return output_video


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SPVC novel-view inference on benchmark MP4 clips."
    )
    parser.add_argument(
        "--benchmark-root",
        "--benchmark_root",
        dest="benchmark_root",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT,
    )
    parser.add_argument("--input-dir", "--input_dir", dest="input_dir", type=Path)
    parser.add_argument(
        "--gt-videos-dir",
        "--gt_videos_dir",
        dest="gt_videos_dir",
        type=Path,
    )
    parser.add_argument(
        "--structured-condition-dir",
        "--structured_condition_dir",
        dest="structured_condition_dir",
        type=Path,
        help="Directory containing combined_<scene>_<start>_0_<direction>_<scale>.mp4.",
    )
    parser.add_argument(
        "--camera-pose-dir",
        "--camera_pose_dir",
        dest="camera_pose_dir",
        type=Path,
        help="Directory containing a camera-pose .pt with the same stem as each control video.",
    )
    parser.add_argument(
        "--output-root", "--output_root", dest="output_root", type=Path
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
        "--camera-pose-encoder-ckpt",
        "--camera_pose_encoder_ckpt",
        dest="camera_pose_encoder_ckpt",
        type=Path,
        help="Checkpoint containing relpose_ecam weights; defaults to --ckpt-high-noise.",
    )
    parser.add_argument(
        "--scenes",
        default="all",
        help="Scene IDs: all, 062, or a comma-separated list such as 001,062.",
    )
    parser.add_argument(
        "--scales",
        default="all",
        help="View scales: all, 4.0, or a comma-separated list.",
    )
    parser.add_argument("--model-id", default="PAI/Wan2.2-Fun-A14B-Control")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--vram-limit", type=float)
    parser.add_argument("--num-frames", type=int, default=25)
    parser.add_argument("--width", type=int, default=DEFAULT_INFER_SIZE[0])
    parser.add_argument("--height", type=int, default=DEFAULT_INFER_SIZE[1])
    parser.add_argument("--output-width", type=int, default=DEFAULT_OUTPUT_SIZE[0])
    parser.add_argument("--output-height", type=int, default=DEFAULT_OUTPUT_SIZE[1])
    parser.add_argument("--fps", type=int, default=10)
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate discovery and pairings without loading the model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.input_dir = args.input_dir or args.benchmark_root / "testset"
    args.gt_videos_dir = (
        args.gt_videos_dir or args.benchmark_root / "gt" / "gt_videos"
    )
    args.structured_condition_dir = (
        args.structured_condition_dir
        or args.benchmark_root / "structured-condition"
    )
    args.camera_pose_dir = (
        args.camera_pose_dir or args.benchmark_root / "camera-pose-condition"
    )
    args.output_root = args.output_root or args.benchmark_root / "results"
    args.camera_pose_encoder_ckpt = (
        args.camera_pose_encoder_ckpt or args.ckpt_high_noise
    )

    if args.num_frames <= 0:
        raise ValueError("--num-frames must be positive")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive")
    if args.output_width <= 0 or args.output_height <= 0:
        raise ValueError("--output-width and --output-height must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    scenes = parse_csv_filter(args.scenes, pad_numeric_to=3)
    scales = parse_csv_filter(args.scales)
    clips = discover_clips(args.input_dir, scenes, scales)
    conditions = validate_and_load_conditions(
        clips=clips,
        gt_videos_dir=args.gt_videos_dir,
        structured_condition_dir=args.structured_condition_dir,
        camera_pose_dir=args.camera_pose_dir,
        num_frames=args.num_frames,
    )

    print(f"Benchmark root:   {args.benchmark_root}")
    print(f"Input directory:  {args.input_dir}")
    print(f"GT videos:        {args.gt_videos_dir}")
    print(f"Structured cond.: {args.structured_condition_dir}")
    print(f"Camera poses:     {args.camera_pose_dir}")
    print(f"Output directory: {args.output_root}")
    print(f"Matched clips:    {len(clips)}")
    print(f"Scenes:           {sorted({clip.scene_id for clip in clips})}")
    print(f"Scales:           {sorted({clip.scale for clip in clips})}")

    if args.dry_run:
        for index, clip in enumerate(clips, start=1):
            paths, _ = conditions[clip.name]
            print_conditions(index, len(clips), paths)
        print("Dry run finished; all selected condition files and camera poses are valid.")
        return

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    for checkpoint in (
        args.ckpt_high_noise,
        args.ckpt_low_noise,
        args.camera_pose_encoder_ckpt,
    ):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print(f"Device:           {device}")
    pipe = build_pipeline(args, device)
    load_camera_pose_encoder(
        pipe,
        args.camera_pose_encoder_ckpt,
        device=device,
        dtype=torch.bfloat16,
    )
    print(f"Camera encoder:   {args.camera_pose_encoder_ckpt}")
    infer_size = (args.width, args.height)
    output_size = (args.output_width, args.output_height)

    for index, clip in enumerate(clips, start=1):
        paths, camera_pose = conditions[clip.name]
        print_conditions(index, len(clips), paths)
        condition_kwargs = build_condition_kwargs(
            paths=paths,
            camera_pose=camera_pose,
            infer_size=infer_size,
            num_frames=args.num_frames,
        )
        with torch.inference_mode():
            video = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                seed=args.seed,
                cfg_scale=args.cfg_scale,
                switch_DiT_boundary=args.switch_dit_boundary,
                num_inference_steps=args.num_inference_steps,
                sigma_shift=args.sigma_shift,
                tiled=args.tiled,
                tile_size=tuple(args.tile_size),
                tile_stride=tuple(args.tile_stride),
                **condition_kwargs,
            )
        output_path = save_outputs(
            frames=video,
            clip=clip,
            output_root=args.output_root,
            output_size=output_size,
            fps=args.fps,
            quality=args.quality,
        )
        print(f"[{index}/{len(clips)}] Saved:     {output_path}")

    print("Novel-view inference finished.")


if __name__ == "__main__":
    main()
