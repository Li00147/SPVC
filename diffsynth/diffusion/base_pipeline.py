from PIL import Image
import torch
import numpy as np
from einops import repeat, reduce
from typing import Union
from ..core.device.npu_compatible_device import parse_device_type
from ..core.loader.config import ModelConfig
from ..core.loader.file import load_state_dict
from ..core.vram.layers import AutoWrappedLinear
from ..utils.lora.general import GeneralLoRALoader
from ..models.model_loader import ModelPool


class PipelineUnit:

    def __init__(
        self,
        seperate_cfg: bool = False,
        take_over: bool = False,
        input_params: tuple[str] = None,
        output_params: tuple[str] = None,
        input_params_posi: dict[str, str] = None,
        input_params_nega: dict[str, str] = None,
        onload_model_names: tuple[str] = None,
    ):
        self.seperate_cfg = seperate_cfg
        self.take_over = take_over
        self.input_params = input_params
        self.output_params = output_params
        self.input_params_posi = input_params_posi
        self.input_params_nega = input_params_nega
        self.onload_model_names = onload_model_names

    def fetch_input_params(self):
        params = []
        if self.input_params is not None:
            for param in self.input_params:
                params.append(param)
        if self.input_params_posi is not None:
            for _, param in self.input_params_posi.items():
                params.append(param)
        if self.input_params_nega is not None:
            for _, param in self.input_params_nega.items():
                params.append(param)
        params = sorted(list(set(params)))
        return params

    def fetch_output_params(self):
        params = []
        if self.output_params is not None:
            for param in self.output_params:
                params.append(param)
        return params

    def process(self, pipe, **kwargs) -> dict:
        return {}

    def post_process(self, pipe, **kwargs) -> dict:
        return {}


class BasePipeline(torch.nn.Module):

    def __init__(
        self,
        device="cuda",
        torch_dtype=torch.float16,
        height_division_factor=64,
        width_division_factor=64,
        time_division_factor=None,
        time_division_remainder=None,
    ):
        super().__init__()
        self.device = device
        self.torch_dtype = torch_dtype
        self.device_type = parse_device_type(device)
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.vram_management_enabled = False
        self.unit_runner = PipelineUnitRunner()
        self.lora_loader = GeneralLoRALoader

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        super().to(*args, **kwargs)
        return self

    def check_resize_height_width(self, height, width, num_frames=None):
        if height % self.height_division_factor != 0:
            height = (
                (height + self.height_division_factor - 1)
                // self.height_division_factor
                * self.height_division_factor
            )
            print(
                f"height % {self.height_division_factor} != 0. We round it up to {height}."
            )
        if width % self.width_division_factor != 0:
            width = (
                (width + self.width_division_factor - 1)
                // self.width_division_factor
                * self.width_division_factor
            )
            print(
                f"width % {self.width_division_factor} != 0. We round it up to {width}."
            )
        if num_frames is None:
            return (height, width)
        else:
            if num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames = (
                    (num_frames + self.time_division_factor - 1)
                    // self.time_division_factor
                    * self.time_division_factor
                    + self.time_division_remainder
                )
                print(
                    f"num_frames % {self.time_division_factor} != {self.time_division_remainder}. We round it up to {num_frames}."
                )
            return (height, width, num_frames)

    def preprocess_image(
        self,
        image,
        torch_dtype=None,
        device=None,
        pattern="B C H W",
        min_value=-1,
        max_value=1,
    ):
        image = torch.Tensor(np.array(image, dtype=np.float32))
        image = image.to(
            dtype=torch_dtype or self.torch_dtype, device=device or self.device
        )
        image = image * ((max_value - min_value) / 255) + min_value
        image = repeat(
            image, f"H W C -> {pattern}", **{"B": 1} if "B" in pattern else {}
        )
        return image

    def preprocess_video(
        self,
        video,
        torch_dtype=None,
        device=None,
        pattern="B C T H W",
        min_value=-1,
        max_value=1,
    ):
        video = [
            self.preprocess_image(
                image,
                torch_dtype=torch_dtype,
                device=device,
                min_value=min_value,
                max_value=max_value,
            )
            for image in video
        ]
        video = torch.stack(video, dim=pattern.index("T") // 2)
        return video

    def vae_output_to_image(
        self, vae_output, pattern="B C H W", min_value=-1, max_value=1
    ):
        if pattern != "H W C":
            vae_output = reduce(vae_output, f"{pattern} -> H W C", reduction="mean")
        image = ((vae_output - min_value) * (255 / (max_value - min_value))).clip(
            0, 255
        )
        image = image.to(device="cpu", dtype=torch.uint8)
        image = Image.fromarray(image.numpy())
        return image

    def vae_output_to_video(
        self, vae_output, pattern="B C T H W", min_value=-1, max_value=1
    ):
        if pattern != "T H W C":
            vae_output = reduce(vae_output, f"{pattern} -> T H W C", reduction="mean")
        video = [
            self.vae_output_to_image(
                image, pattern="H W C", min_value=min_value, max_value=max_value
            )
            for image in vae_output
        ]
        return video

    def load_models_to_device(self, model_names):
        if self.vram_management_enabled:
            for name, model in self.named_children():
                if name not in model_names:
                    if (
                        hasattr(model, "vram_management_enabled")
                        and model.vram_management_enabled
                    ):
                        if hasattr(model, "offload"):
                            model.offload()
                        else:
                            for module in model.modules():
                                if hasattr(module, "offload"):
                                    module.offload()
            getattr(torch, self.device_type).empty_cache()
            for name, model in self.named_children():
                if name in model_names:
                    if (
                        hasattr(model, "vram_management_enabled")
                        and model.vram_management_enabled
                    ):
                        if hasattr(model, "onload"):
                            model.onload()
                        else:
                            for module in model.modules():
                                if hasattr(module, "onload"):
                                    module.onload()

    def generate_noise(
        self,
        shape,
        seed=None,
        rand_device="cpu",
        rand_torch_dtype=torch.float32,
        device=None,
        torch_dtype=None,
    ):
        generator = (
            None if seed is None else torch.Generator(rand_device).manual_seed(seed)
        )
        noise = torch.randn(
            shape, generator=generator, device=rand_device, dtype=rand_torch_dtype
        )
        noise = noise.to(
            dtype=torch_dtype or self.torch_dtype, device=device or self.device
        )
        return noise

    def get_module(self, model, name):
        if "." in name:
            name, suffix = (name[: name.index(".")], name[name.index(".") + 1 :])
            if name.isdigit():
                return self.get_module(model[int(name)], suffix)
            else:
                return self.get_module(getattr(model, name), suffix)
        else:
            return getattr(model, name)

    def freeze_except(self, model_names):
        self.eval()
        self.requires_grad_(False)
        for name in model_names:
            module = self.get_module(self, name)
            if module is None:
                print(
                    f"No {name} models in the pipeline. We cannot enable training on the model. If this occurs during the data processing stage, it is normal."
                )
                continue
            module.train()
            module.requires_grad_(True)

    def load_lora(
        self,
        module: torch.nn.Module,
        lora_config: Union[ModelConfig, str] = None,
        alpha=1,
        hotload=None,
        state_dict=None,
    ):
        if state_dict is None:
            if isinstance(lora_config, str):
                lora = load_state_dict(
                    lora_config, torch_dtype=self.torch_dtype, device=self.device
                )
            else:
                lora_config.download_if_necessary()
                lora = load_state_dict(
                    lora_config.path, torch_dtype=self.torch_dtype, device=self.device
                )
        else:
            lora = state_dict
        lora_loader = self.lora_loader(torch_dtype=self.torch_dtype, device=self.device)
        lora = lora_loader.convert_state_dict(lora)
        if hotload is None:
            hotload = hasattr(module, "vram_management_enabled") and getattr(
                module, "vram_management_enabled"
            )
        if hotload:
            if not (
                hasattr(module, "vram_management_enabled")
                and getattr(module, "vram_management_enabled")
            ):
                raise ValueError(
                    "VRAM Management is not enabled. LoRA hotloading is not supported."
                )
            updated_num = 0
            for _, module in module.named_modules():
                if isinstance(module, AutoWrappedLinear):
                    name = module.name
                    lora_a_name = f"{name}.lora_A.weight"
                    lora_b_name = f"{name}.lora_B.weight"
                    if lora_a_name in lora and lora_b_name in lora:
                        updated_num += 1
                        module.lora_A_weights.append(lora[lora_a_name] * alpha)
                        module.lora_B_weights.append(lora[lora_b_name])
            print(
                f"{updated_num} tensors are patched by LoRA. You can use `pipe.clear_lora()` to clear all LoRA layers."
            )
        else:
            lora_loader.fuse_lora_to_base_model(module, lora, alpha=alpha)

    def download_and_load_models(
        self, model_configs: list[ModelConfig] = [], vram_limit: float = None
    ):
        model_pool = ModelPool()
        for model_config in model_configs:
            model_config.download_if_necessary()
            vram_config = model_config.vram_config()
            vram_config["computation_dtype"] = (
                vram_config["computation_dtype"] or self.torch_dtype
            )
            vram_config["computation_device"] = (
                vram_config["computation_device"] or self.device
            )
            model_pool.auto_load_model(
                model_config.path,
                vram_config=vram_config,
                vram_limit=vram_limit,
                clear_parameters=model_config.clear_parameters,
            )
        return model_pool

    def check_vram_management_state(self):
        vram_management_enabled = False
        for module in self.children():
            if hasattr(module, "vram_management_enabled") and getattr(
                module, "vram_management_enabled"
            ):
                vram_management_enabled = True
        return vram_management_enabled


class PipelineUnitRunner:

    def __init__(self):
        pass

    def __call__(
        self,
        unit: PipelineUnit,
        pipe: BasePipeline,
        inputs_shared: dict,
        inputs_posi: dict,
        inputs_nega: dict,
    ) -> tuple[dict, dict]:
        if unit.take_over:
            inputs_shared, inputs_posi, inputs_nega = unit.process(
                pipe,
                inputs_shared=inputs_shared,
                inputs_posi=inputs_posi,
                inputs_nega=inputs_nega,
            )
        elif unit.seperate_cfg:
            processor_inputs = {
                name: inputs_posi.get(name_)
                for name, name_ in unit.input_params_posi.items()
            }
            if unit.input_params is not None:
                for name in unit.input_params:
                    processor_inputs[name] = inputs_shared.get(name)
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_posi.update(processor_outputs)
            if inputs_shared["cfg_scale"] != 1:
                processor_inputs = {
                    name: inputs_nega.get(name_)
                    for name, name_ in unit.input_params_nega.items()
                }
                if unit.input_params is not None:
                    for name in unit.input_params:
                        processor_inputs[name] = inputs_shared.get(name)
                processor_outputs = unit.process(pipe, **processor_inputs)
                inputs_nega.update(processor_outputs)
            else:
                inputs_nega.update(processor_outputs)
        else:
            processor_inputs = {
                name: inputs_shared.get(name) for name in unit.input_params
            }
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_shared.update(processor_outputs)
        return (inputs_shared, inputs_posi, inputs_nega)
