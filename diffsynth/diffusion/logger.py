import os, torch
from datetime import datetime
from accelerate import Accelerator


class ModelLogger:

    def __init__(
        self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x: x
    ):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        os.makedirs(self.output_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.loss_file_path = os.path.join(self.output_path, f"loss_{timestamp}.txt")
        with open(self.loss_file_path, "w") as f:
            f.write(
                f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write("Step,Loss\n")

    def on_step_end(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        save_steps=None,
        loss=None,
    ):
        self.num_steps += 1
        if loss is not None and accelerator.is_main_process:
            loss_value = loss.item() if isinstance(loss, torch.Tensor) else loss
            with open(self.loss_file_path, "a") as f:
                f.write(f"{self.num_steps},{loss_value}\n")
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(
                state_dict, remove_prefix=self.remove_prefix_in_ckpt
            )
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)

    def on_training_end(
        self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None
    ):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(
                state_dict, remove_prefix=self.remove_prefix_in_ckpt
            )
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
