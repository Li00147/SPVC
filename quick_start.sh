
python scripts/inference.py \
    --control_video demo/control-video.mp4 \
    --reference_video demo/reference-video.mp4 \
    --hdmap_bbox_video demo/structured-condition.mp4 \
    --cam_pose demo/camera-pose.pt \
    --lora_ckpt_high checkpoints/spvc_high_noise.safetensors \
    --lora_ckpt_low checkpoints/spvc_low_noise.safetensors \
    --height 464 \
    --width 800 \
    --num_frames 25 \
    --fps 12 \
    --output demo/result_right_1.0_wo_cam_wo_cond_wo_ref.mp4


python scripts/inference.py \
    --control_video demo/Panoptic-Fix/control-video.mp4 \
    --reference_video demo/Panoptic-Fix/reference-video.mp4 \
    --lora_ckpt_high checkpoints/spvc_high_noise.safetensors \
    --lora_ckpt_low checkpoints/spvc_low_noise.safetensors \
    --height 464 \
    --width 800 \
    --num_frames 25 \
    --fps 12 \
    --output demo/Panoptic-Fix/fixed-video.mp4