import json

import torch
from peft import LoraConfig, inject_adapter_in_model

from ..core.loader.config import ModelConfig
from ..core.loader.file import load_state_dict


class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def to(self, *args, **kwargs):
        for model in self.children():
            model.to(*args, **kwargs)
        return self

    def trainable_modules(self):
        return filter(lambda parameter: parameter.requires_grad, self.parameters())

    def trainable_param_names(self):
        return {
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_names = self.trainable_param_names()
        state_dict = {
            name: parameter
            for name, parameter in state_dict.items()
            if name in trainable_names
        }
        if remove_prefix is None:
            return state_dict
        return {
            (
                name[len(remove_prefix) :] if name.startswith(remove_prefix) else name
            ): value
            for name, value in state_dict.items()
        }

    def transfer_data_to_device(self, data, device, torch_float_dtype=None):
        if isinstance(data, torch.Tensor):
            data = data.to(device)
            if torch_float_dtype is not None and data.is_floating_point():
                data = data.to(torch_float_dtype)
            return data
        if isinstance(data, tuple):
            return tuple(
                self.transfer_data_to_device(item, device, torch_float_dtype)
                for item in data
            )
        if isinstance(data, list):
            return [
                self.transfer_data_to_device(item, device, torch_float_dtype)
                for item in data
            ]
        if isinstance(data, dict):
            return {
                key: self.transfer_data_to_device(value, device, torch_float_dtype)
                for key, value in data.items()
            }
        return data

    def parse_model_configs(
        self, model_paths, model_id_with_origin_paths, device="cpu"
    ):
        configs = []
        if model_paths is not None:
            configs.extend(ModelConfig(path=path) for path in json.loads(model_paths))
        if model_id_with_origin_paths is not None:
            for value in model_id_with_origin_paths.split(","):
                model_id, origin_file_pattern = value.split(":", 1)
                configs.append(
                    ModelConfig(
                        model_id=model_id,
                        origin_file_pattern=origin_file_pattern,
                    )
                )
        return configs

    def switch_pipe_to_training_mode(
        self,
        pipe,
        lora_base_model,
        lora_target_modules,
        lora_rank,
        lora_checkpoint=None,
    ):
        pipe.scheduler.set_timesteps(1000, training=True)
        pipe.freeze_except([])
        if lora_base_model is None:
            raise ValueError("lora_base_model is required")
        model = getattr(pipe, lora_base_model)
        config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(config, model)
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.to(pipe.torch_dtype)
        if lora_checkpoint is not None:
            state_dict = {}
            for key, value in load_state_dict(lora_checkpoint).items():
                if "lora_A.weight" in key or "lora_B.weight" in key:
                    key = key.replace("lora_A.weight", "lora_A.default.weight")
                    key = key.replace("lora_B.weight", "lora_B.default.weight")
                    state_dict[key] = value
                elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                    state_dict[key] = value
            model.load_state_dict(state_dict, strict=False)
        setattr(pipe, lora_base_model, model)
