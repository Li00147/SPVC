from typing import Optional, Union
import torch
from einops import rearrange
from PIL import Image
from tqdm import tqdm
from typing_extensions import Literal
from ..core.loader.config import ModelConfig
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..diffusion.flow_match import FlowMatchScheduler
from ..models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from ..models.wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder
from ..models.wan_video_vae import WanVideoVAE


class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=16,
            width_division_factor=16,
            time_division_factor=4,
            time_division_remainder=1,
        )
        self.scheduler = FlowMatchScheduler()
        self.tokenizer: Optional[HuggingfaceTokenizer] = None
        self.text_encoder: Optional[WanTextEncoder] = None
        self.dit: Optional[WanModel] = None
        self.dit2: Optional[WanModel] = None
        self.vae: Optional[WanVideoVAE] = None
        self.relpose_ecam = None
        self.in_iteration_models = ("dit",)
        self.in_iteration_models_2 = ("dit2",)
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_RelPoseCondition(),
            WanVideoUnit_Control(),
            WanVideoUnit_CombinedVideoReference(),
            WanVideoUnit_ReferenceVideo(),
            WanVideoUnit_CfgMerger(),
        ]
        self.model_fn = model_fn_wan_video

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: Optional[list[ModelConfig]] = None,
        tokenizer_config: Optional[ModelConfig] = None,
        vram_limit: Optional[float] = None,
    ):
        if tokenizer_config is None:
            tokenizer_config = ModelConfig(
                model_id="Wan-AI/Wan2.1-T2V-1.3B",
                origin_file_pattern="google/umt5-xxl/",
            )
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs or [], vram_limit)
        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
        dit = model_pool.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_pool.fetch_model("wan_video_vae")
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2
        tokenizer_config.download_if_necessary()
        pipe.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_config.path, seq_len=512, clean="whitespace"
        )
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        input_video: Optional[list[Image.Image]] = None,
        control_video: Optional[list[Image.Image]] = None,
        reference_combined_video: Optional[list[Image.Image]] = None,
        reference_video: Optional[list[Image.Image]] = None,
        cam_pose: Optional[torch.Tensor] = None,
        denoising_strength: float = 1.0,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        cfg_scale: float = 5.0,
        cfg_merge: bool = False,
        switch_DiT_boundary: float = 0.875,
        num_inference_steps: int = 50,
        sigma_shift: float = 5.0,
        tiled: bool = True,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
        progress_bar_cmd=tqdm,
        output_type: Literal["quantized", "floatpoint"] = "quantized",
    ):
        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=denoising_strength,
            shift=sigma_shift,
        )
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "input_video": input_video,
            "control_video": control_video,
            "reference_combined_video": reference_combined_video,
            "reference_video": reference_video,
            "cam_pose": cam_pose,
            "seed": seed,
            "rand_device": rand_device,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "cfg_scale": cfg_scale,
            "cfg_merge": cfg_merge,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega
            )
        self.load_models_to_device(self.in_iteration_models)
        models = {"dit": self.dit}
        for progress_id, timestep in enumerate(
            progress_bar_cmd(self.scheduler.timesteps)
        ):
            if (
                timestep.item() < switch_DiT_boundary * 1000
                and self.dit2 is not None
                and (models["dit"] is not self.dit2)
            ):
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2
            timestep = timestep.unsqueeze(0).to(
                dtype=self.torch_dtype, device=self.device
            )
            noise_pred_posi = self.model_fn(
                **models, **inputs_shared, **inputs_posi, timestep=timestep
            )
            if cfg_scale == 1.0:
                noise_pred = noise_pred_posi
            elif cfg_merge:
                noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                noise_pred = noise_pred_nega + cfg_scale * (
                    noise_pred_posi - noise_pred_nega
                )
            else:
                noise_pred_nega = self.model_fn(
                    **models, **inputs_shared, **inputs_nega, timestep=timestep
                )
                noise_pred = noise_pred_nega + cfg_scale * (
                    noise_pred_posi - noise_pred_nega
                )
            inputs_shared["latents"] = self.scheduler.step(
                noise_pred,
                self.scheduler.timesteps[progress_id],
                inputs_shared["latents"],
            )
        self.load_models_to_device(("vae",))
        video = self.vae.decode(
            inputs_shared["latents"],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        if output_type == "quantized":
            video = self.vae_output_to_video(video)
        self.load_models_to_device(())
        return video


class WanVideoUnit_ShapeChecker(PipelineUnit):

    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(
            height, width, num_frames
        )
        return {"height": height, "width": width, "num_frames": num_frames}


class WanVideoUnit_NoiseInitializer(PipelineUnit):

    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device"),
            output_params=("noise",),
        )

    def process(self, pipe, height, width, num_frames, seed, rand_device):
        length = (num_frames - 1) // 4 + 1
        shape = (
            1,
            pipe.vae.model.z_dim,
            length,
            height // pipe.vae.upsampling_factor,
            width // pipe.vae.upsampling_factor,
        )
        return {"noise": pipe.generate_noise(shape, seed=seed, rand_device=rand_device)}


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):

    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",),
        )

    def process(self, pipe, input_video, noise, tiled, tile_size, tile_stride):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(self.onload_model_names)
        input_video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(
            input_video,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        return {
            "latents": pipe.scheduler.add_noise(
                input_latents, noise, timestep=pipe.scheduler.timesteps[0]
            )
        }


class WanVideoUnit_PromptEmbedder(PipelineUnit):

    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            output_params=("context",),
            onload_model_names=("text_encoder",),
        )

    def process(self, pipe, prompt):
        pipe.load_models_to_device(self.onload_model_names)
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        prompt_emb = pipe.text_encoder(ids, mask)
        for seq_id, seq_len in enumerate(mask.gt(0).sum(dim=1).long()):
            prompt_emb[seq_id, seq_len:] = 0
        return {"context": prompt_emb}


class WanVideoUnit_Control(PipelineUnit):

    def __init__(self):
        super().__init__(
            input_params=(
                "control_video",
                "num_frames",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
                "y",
                "latents",
            ),
            output_params=("y",),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe,
        control_video,
        num_frames,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
        y,
        latents,
    ):
        if control_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        control_latents = pipe.vae.encode(
            pipe.preprocess_video(control_video),
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        y_dim = pipe.dit.in_dim - control_latents.shape[1] - latents.shape[1]
        if y is None:
            y = torch.zeros(
                (1, y_dim, (num_frames - 1) // 4 + 1, height // 8, width // 8),
                dtype=pipe.torch_dtype,
                device=pipe.device,
            )
        else:
            y = y[:, -y_dim:]
        return {"y": torch.cat((control_latents, y), dim=1)}


class WanVideoUnit_RelPoseCondition(PipelineUnit):

    def __init__(self):
        super().__init__(
            input_params=("cam_pose",),
            output_params=("pose_ctx_tokens_raw",),
            onload_model_names=(),
        )

    def process(self, pipe, cam_pose):
        if cam_pose is None:
            return {}
        encoder = getattr(pipe, "relpose_ecam", None)
        if encoder is None:
            encoder = getattr(pipe.dit, "relpose_ecam", None)
        if encoder is None:
            raise RuntimeError(
                "CameraPoseEncoder is not attached to the pipeline or DiT"
            )
        cam_pose = cam_pose.to(device=pipe.device, dtype=pipe.torch_dtype)
        return {"pose_ctx_tokens_raw": encoder(cam_pose)}


class WanVideoUnit_CombinedVideoReference(PipelineUnit):

    def __init__(self):
        super().__init__(
            input_params=("reference_combined_video",),
            output_params=("reference_combined_video_latents",),
            onload_model_names=("vae",),
        )

    def process(self, pipe, reference_combined_video):
        if reference_combined_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        latents = pipe.vae.encode(
            pipe.preprocess_video(reference_combined_video), device=pipe.device
        )
        return {
            "reference_combined_video_latents": latents.to(
                dtype=pipe.torch_dtype, device=pipe.device
            )
        }


class WanVideoUnit_ReferenceVideo(PipelineUnit):

    def __init__(self):
        super().__init__(
            input_params=("reference_video",),
            output_params=("reference_video_latents",),
            onload_model_names=("vae",),
        )

    def process(self, pipe, reference_video):
        if reference_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        latents = pipe.vae.encode(
            pipe.preprocess_video(reference_video), device=pipe.device
        )
        return {
            "reference_video_latents": latents.to(
                dtype=pipe.torch_dtype, device=pipe.device
            )
        }


class WanVideoUnit_CfgMerger(PipelineUnit):

    def __init__(self):
        super().__init__(take_over=True)
        self.tensor_names = (
            "context",
            "y",
            "reference_combined_video_latents",
            "reference_video_latents",
        )

    def process(self, pipe, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return (inputs_shared, inputs_posi, inputs_nega)
        for name in self.tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.cat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.cat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return (inputs_shared, inputs_posi, inputs_nega)


def _reference_tokens(dit, latents, batch_size):
    if latents is None or latents.ndim != 5:
        return None
    tokens = torch.cat(
        [
            dit.ref_conv(latents[:, :, frame_id]).flatten(2).transpose(1, 2)
            for frame_id in range(latents.shape[2])
        ],
        dim=1,
    )
    if tokens.shape[0] != batch_size:
        tokens = tokens.repeat(batch_size // tokens.shape[0], 1, 1)
    return tokens


def _rope_frequencies(dit, frames, height, width):
    return torch.cat(
        (
            dit.freqs[0][:frames]
            .view(frames, 1, 1, -1)
            .expand(frames, height, width, -1),
            dit.freqs[1][:height]
            .view(1, height, 1, -1)
            .expand(frames, height, width, -1),
            dit.freqs[2][:width]
            .view(1, 1, width, -1)
            .expand(frames, height, width, -1),
        ),
        dim=-1,
    ).reshape(frames * height * width, 1, -1)


def model_fn_wan_video(
    dit: WanModel,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    reference_combined_video_latents: Optional[torch.Tensor] = None,
    reference_video_latents: Optional[torch.Tensor] = None,
    pose_ctx_tokens_raw: Optional[torch.Tensor] = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    cond_drop_rate: float = 0.0,
    **kwargs
):
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)
    x = latents
    if x.shape[0] != context.shape[0]:
        x = x.repeat(context.shape[0] // x.shape[0], 1, 1, 1, 1)
    if y is None:
        raise ValueError("control_video is required")
    x = torch.cat((x, y), dim=1)
    x = dit.patchify(x)
    frames, height, width = x.shape[2:]
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
    if pose_ctx_tokens_raw is not None:
        if pose_ctx_tokens_raw.shape[1] != frames:
            indices = (
                torch.linspace(
                    0,
                    pose_ctx_tokens_raw.shape[1] - 1,
                    steps=frames,
                    device=pose_ctx_tokens_raw.device,
                )
                .round()
                .long()
            )
            pose_ctx_tokens_raw = pose_ctx_tokens_raw[:, indices]
        if pose_ctx_tokens_raw.shape[0] != context.shape[0]:
            pose_ctx_tokens_raw = pose_ctx_tokens_raw.repeat(
                context.shape[0] // pose_ctx_tokens_raw.shape[0], 1, 1
            )
        context = torch.cat((pose_ctx_tokens_raw, context), dim=1)
    if cond_drop_rate > 0 and torch.rand(1).item() < cond_drop_rate:
        reference_combined_video_latents = None
    reference_tensors = []
    reference_frame_counts = []
    for reference_latents in (
        reference_video_latents,
        reference_combined_video_latents,
    ):
        tokens = _reference_tokens(dit, reference_latents, x.shape[0])
        if tokens is not None:
            reference_tensors.append(tokens)
            reference_frame_counts.append(reference_latents.shape[2])
    if reference_tensors:
        x = torch.cat((*reference_tensors, x), dim=1)
    frequencies = [
        _rope_frequencies(dit, count, height, width) for count in reference_frame_counts
    ]
    frequencies.append(_rope_frequencies(dit, frames, height, width))
    freqs = torch.cat(frequencies, dim=0).to(x.device)

    def checkpointed_forward(module, hidden_states):

        def custom_forward(*inputs):
            return module(*inputs)

        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                return torch.utils.checkpoint.checkpoint(
                    custom_forward,
                    hidden_states,
                    context,
                    t_mod,
                    freqs,
                    use_reentrant=False,
                )
        if use_gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                context,
                t_mod,
                freqs,
                use_reentrant=False,
            )
        return module(hidden_states, context, t_mod, freqs)

    for block in dit.blocks:
        x = checkpointed_forward(block, x)
    x = dit.head(x, t)
    reference_token_count = sum((tensor.shape[1] for tensor in reference_tensors))
    if reference_token_count:
        x = x[:, reference_token_count:]
    return dit.unpatchify(x, (frames, height, width))
