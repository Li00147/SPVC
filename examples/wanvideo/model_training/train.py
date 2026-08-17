import argparse
import os
import warnings

import accelerate
import torch

from diffsynth.core.data.unified_dataset import UnifiedDataset
from diffsynth.core.loader.config import ModelConfig
from diffsynth.diffusion.logger import ModelLogger
from diffsynth.diffusion.loss import FlowMatchSFTLoss
from diffsynth.diffusion.parsers import add_general_config, add_video_size_config
from diffsynth.diffusion.runner import launch_training_task
from diffsynth.diffusion.training_module import DiffusionTrainingModule
from diffsynth.models.camera_pose_encoder import CameraPoseEncoder
from diffsynth.pipelines.wan_video import WanVideoPipeline

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanTrainingModule(DiffusionTrainingModule):

    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        device="cpu",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        enable_cond_drop=False,
        max_training_step=None,
        resume_step=0,
    ):
        super().__init__()
        if not use_gradient_checkpointing:
            warnings.warn(
                "Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing."
            )
            use_gradient_checkpointing = True
        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            device=device,
        )
        tokenizer_config = (
            ModelConfig(
                model_id="Wan-AI/Wan2.1-T2V-1.3B",
                origin_file_pattern="google/umt5-xxl/",
            )
            if tokenizer_path is None
            else ModelConfig(tokenizer_path)
        )
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
        )
        self.switch_pipe_to_training_mode(
            self.pipe,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint,
        )
        if "cam_pose" in (extra_inputs or ""):
            dim = self.pipe.dit.dim if hasattr(self.pipe.dit, "dim") else 5120
            self.relpose_ecam = CameraPoseEncoder(dim=dim, num_freqs=6, hidden=1024).to(
                device=device, dtype=torch.bfloat16
            )
            self.pipe.relpose_ecam = self.relpose_ecam
            if hasattr(self.pipe, "dit") and self.pipe.dit is not None:
                self.pipe.dit.relpose_ecam = self.relpose_ecam
            if hasattr(self.pipe, "dit2") and self.pipe.dit2 is not None:
                self.pipe.dit2.relpose_ecam = self.relpose_ecam
            for p in self.relpose_ecam.parameters():
                p.requires_grad = True
            if "relpose_ecam" not in self.pipe.in_iteration_models:
                self.pipe.in_iteration_models = self.pipe.in_iteration_models + (
                    "relpose_ecam",
                )
            if lora_checkpoint is not None:
                from diffsynth.core.loader.file import load_state_dict

                ckpt_sd = load_state_dict(lora_checkpoint)
                sub = {}
                for k, v in ckpt_sd.items():
                    if k.startswith("pipe.relpose_ecam."):
                        sub[k[len("pipe.relpose_ecam.") :]] = v
                    elif k.startswith("relpose_ecam."):
                        sub[k[len("relpose_ecam.") :]] = v
                if sub:
                    load_result = self.relpose_ecam.load_state_dict(sub, strict=False)
                    if load_result.missing_keys:
                        warnings.warn(
                            f"Missing CameraPoseEncoder keys: {load_result.missing_keys}"
                        )
                    if load_result.unexpected_keys:
                        warnings.warn(
                            f"Unexpected CameraPoseEncoder keys: {load_result.unexpected_keys}"
                        )
        else:
            self.relpose_ecam = None
            self.pipe.relpose_ecam = None
            if hasattr(self.pipe, "dit") and self.pipe.dit is not None:
                self.pipe.dit.relpose_ecam = None
            if hasattr(self.pipe, "dit2") and self.pipe.dit2 is not None:
                self.pipe.dit2.relpose_ecam = None
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.enable_cond_drop = enable_cond_drop
        self.max_training_step = max_training_step
        self.total_training_steps = 0
        self.current_step = resume_step

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            inputs_shared[extra_input] = data[extra_input]
        return inputs_shared

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return (inputs_shared, inputs_posi, inputs_nega)

    def forward(self, data, inputs=None):
        self.current_step += 1
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        if self.enable_cond_drop:
            max_step = (
                self.max_training_step
                if self.max_training_step is not None
                else self.total_training_steps
            )
            if max_step > 0:
                cond_drop_rate = min(self.current_step / max_step, 1.0)
                inputs[0]["cond_drop_rate"] = cond_drop_rate
        inputs = self.transfer_data_to_device(
            inputs, self.pipe.device, self.pipe.torch_dtype
        )
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        return FlowMatchSFTLoss(self.pipe, **inputs[0], **inputs[1])


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument(
        "--tokenizer_path", type=str, default=None, help="Path to tokenizer."
    )
    parser.add_argument(
        "--max_timestep_boundary",
        type=float,
        default=1.0,
        help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).",
    )
    parser.add_argument(
        "--min_timestep_boundary",
        type=float,
        default=0.0,
        help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).",
    )
    parser.add_argument(
        "--enable_cond_drop",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable condition dropout: drop rate linearly increases from 0 to 1 over total training steps.",
    )
    parser.add_argument(
        "--max_training_step",
        type=int,
        default=None,
        help="Manually set the max training step to override the auto-calculated total_training_steps for cond_drop scheduling.",
    )
    parser.add_argument(
        "--resume_step",
        type=int,
        default=0,
        help="Resume training from this step count, so that cond_drop_rate continues from the correct position.",
    )
    parser.add_argument(
        "--initialize_model_on_cpu",
        default=False,
        action="store_true",
        help="Whether to initialize models on CPU.",
    )
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters
            )
        ],
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        enable_cond_drop=args.enable_cond_drop,
        max_training_step=args.max_training_step,
        resume_step=args.resume_step,
    )
    model_logger = ModelLogger(
        args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    launch_training_task(accelerator, dataset, model, model_logger, args=args)
