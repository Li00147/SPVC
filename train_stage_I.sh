#!/usr/bin/env bash
set -euo pipefail

SPVC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SPVC_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

SPVC_STAGE_I_DATA_ROOT="${SPVC_STAGE_I_DATA_ROOT:-/data01/lg/OmniFix/stage-I-train-data/processed}"
SPVC_STAGE_I_METADATA="${SPVC_STAGE_I_METADATA:-${SPVC_STAGE_I_DATA_ROOT}/dataset_ref_video_cam_pose.json}"
SPVC_OUTPUT_ROOT="${SPVC_OUTPUT_ROOT:-${SPVC_ROOT}/models/train}"
SPVC_NUM_FRAMES="${SPVC_NUM_FRAMES:-25}"
ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-${SPVC_ROOT}/accelerate_config.yaml}"

SPVC_ACCELERATE=(accelerate launch --config_file "${ACCELERATE_CONFIG_FILE}")

SPVC_MODEL_COMMON="PAI/Wan2.2-Fun-A14B-Control:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.2-Fun-A14B-Control:Wan2.1_VAE.pth"
SPVC_COMMON_ARGS=(
  --dataset_base_path "${SPVC_STAGE_I_DATA_ROOT}"
  --dataset_metadata_path "${SPVC_STAGE_I_METADATA}"
  --data_file_keys "video,control_video,reference_video,cam_pose"
  --height 448
  --width 800
  --num_frames "${SPVC_NUM_FRAMES}"
  --dataset_repeat 10
  --learning_rate 1e-4
  --num_epochs 2
  --remove_prefix_in_ckpt "pipe.dit."
  --lora_base_model "dit"
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2"
  --lora_rank 64
  --save_steps 150
  --extra_inputs "control_video,reference_video,cam_pose"
)

"${SPVC_ACCELERATE[@]}" examples/wanvideo/model_training/train.py \
  "${SPVC_COMMON_ARGS[@]}" \
  --model_id_with_origin_paths "PAI/Wan2.2-Fun-A14B-Control:high_noise_model/diffusion_pytorch_model*.safetensors,${SPVC_MODEL_COMMON}" \
  --output_path "${SPVC_OUTPUT_ROOT}/Wan2.2-Fun-A14B-Control_high_noise_stage_I" \
  --max_timestep_boundary 0.358 \
  --min_timestep_boundary 0

"${SPVC_ACCELERATE[@]}" examples/wanvideo/model_training/train.py \
  "${SPVC_COMMON_ARGS[@]}" \
  --model_id_with_origin_paths "PAI/Wan2.2-Fun-A14B-Control:low_noise_model/diffusion_pytorch_model*.safetensors,${SPVC_MODEL_COMMON}" \
  --output_path "${SPVC_OUTPUT_ROOT}/Wan2.2-Fun-A14B-Control_low_noise_stage_I" \
  --max_timestep_boundary 1 \
  --min_timestep_boundary 0.358
