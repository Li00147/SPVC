#!/usr/bin/env bash

set -euo pipefail

HIGH_NOISE_CKPT="${SPVC_HIGH_NOISE_CKPT:-checkpoints/spvc_high_noise.safetensors}"
LOW_NOISE_CKPT="${SPVC_LOW_NOISE_CKPT:-checkpoints/spvc_low_noise.safetensors}"
CAMERA_POSE_CKPT="${SPVC_CAMERA_POSE_CKPT:-$HIGH_NOISE_CKPT}"

python scripts/inference.py \
    --control_video demo/NVS-Fix/control-video.mp4 \
    --reference_video demo/NVS-Fix/reference-video.mp4 \
    --hdmap_bbox_video demo/NVS-Fix/structured-condition.mp4 \
    --cam_pose demo/NVS-Fix/camera-pose.pt \
    --cam_pose_ckpt "$CAMERA_POSE_CKPT" \
    --lora_ckpt_high "$HIGH_NOISE_CKPT" \
    --lora_ckpt_low "$LOW_NOISE_CKPT" \
    --height 464 \
    --width 800 \
    --num_frames 25 \
    --fps 12 \
    --output demo/NVS-Fix/fixed-video.mp4

python scripts/inference.py \
    --control_video demo/Panoptic-Fix/control-video.mp4 \
    --reference_video demo/Panoptic-Fix/reference-video.mp4 \
    --lora_ckpt_high "$HIGH_NOISE_CKPT" \
    --lora_ckpt_low "$LOW_NOISE_CKPT" \
    --height 464 \
    --width 800 \
    --num_frames 25 \
    --fps 12 \
    --output demo/Panoptic-Fix/fixed-video.mp4
