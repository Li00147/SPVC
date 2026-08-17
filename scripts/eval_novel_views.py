import argparse
from pathlib import Path

import numpy as np
import torch

from diffsynth.core.loader.config import ModelConfig
from diffsynth.models.camera_pose_encoder import load_camera_pose_encoder
from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.utils.data import VideoData, save_video


DEFAULT_PROMPT = "Transform a low-quality autonomous driving video into a high-quality, realistic driving video with clear details and consistent motion."
DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


def load_video_frames(path, height, width, num_frames, name):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    video = VideoData(str(path), height=height, width=width)
    if len(video) < num_frames:
        raise ValueError(
            f"{name} has {len(video)} frames, but {num_frames} frames are required"
        )
    return [video[index] for index in range(num_frames)]


def load_camera_pose(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Camera pose not found: {path}")
    if path.suffix.lower() == ".npy":
        pose = torch.from_numpy(np.load(path))
    else:
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
            f"Camera pose must have shape (F,4,4) or (B,F,4,4), got {tuple(pose.shape)}"
        )
    return pose


def build_pipeline(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    vram_config = {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cpu",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": args.device,
        "computation_dtype": torch.bfloat16,
        "computation_device": args.device,
    }
    model_patterns = (
        "high_noise_model/diffusion_pytorch_model*.safetensors",
        "low_noise_model/diffusion_pytorch_model*.safetensors",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth",
    )
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
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
    pipe.load_lora(pipe.dit, args.lora_ckpt_high, alpha=args.lora_alpha)
    pipe.load_lora(pipe.dit2, args.lora_ckpt_low, alpha=args.lora_alpha)
    return pipe


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SPVC inference with control, HD-map/3D-bbox, reference-video, and camera-pose conditions."
    )
    parser.add_argument("--control_video", required=True)
    parser.add_argument(
        "--hdmap_bbox_video",
        "--reference_combined_video",
        dest="hdmap_bbox_video",
        required=True,
    )
    parser.add_argument("--reference_video", required=True)
    parser.add_argument("--cam_pose")
    parser.add_argument("--cam_pose_ckpt")
    parser.add_argument("--lora_ckpt_high", required=True)
    parser.add_argument("--lora_ckpt_low", required=True)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--output", default="outputs/spvc.mp4")
    parser.add_argument("--model_id", default="PAI/Wan2.2-Fun-A14B-Control")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=464)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--num_frames", type=int, default=25)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--switch_dit_boundary", type=float, default=0.875)
    parser.add_argument("--vram_limit", type=float)
    parser.add_argument("--tiled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tile_size", type=int, nargs=2, default=(30, 52))
    parser.add_argument("--tile_stride", type=int, nargs=2, default=(15, 26))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    control_video = load_video_frames(
        args.control_video,
        args.height,
        args.width,
        args.num_frames,
        "Control video",
    )
    hdmap_bbox_video = load_video_frames(
        args.hdmap_bbox_video,
        args.height,
        args.width,
        args.num_frames,
        "HD-map/3D-bbox video",
    )
    reference_video = load_video_frames(
        args.reference_video,
        args.height,
        args.width,
        args.num_frames,
        "Reference video",
    )
    cam_pose = load_camera_pose(args.cam_pose) if args.cam_pose else None
    pipe = build_pipeline(args)
    if cam_pose is not None:
        checkpoint = args.cam_pose_ckpt or args.lora_ckpt_high
        load_camera_pose_encoder(
            pipe,
            checkpoint,
            device=args.device,
            dtype=torch.bfloat16,
        )
    print(f"control_video: {args.control_video}")
    print(f"hdmap_bbox_video: {args.hdmap_bbox_video}")
    print(f"reference_video: {args.reference_video}")
    if args.cam_pose:
        print(f"cam_pose: {args.cam_pose} {tuple(cam_pose.shape)}")
    with torch.inference_mode():
        video = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            control_video=control_video,
            reference_combined_video=hdmap_bbox_video,
            reference_video=reference_video,
            cam_pose=cam_pose,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            cfg_scale=args.cfg_scale,
            switch_DiT_boundary=args.switch_dit_boundary,
            num_inference_steps=args.num_inference_steps,
            sigma_shift=args.sigma_shift,
            seed=args.seed,
            tiled=args.tiled,
            tile_size=tuple(args.tile_size),
            tile_stride=tuple(args.tile_stride),
        )
    save_video(video, str(output_path), fps=args.fps, quality=5)
    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    main()
