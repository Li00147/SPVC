import math
from pathlib import Path

import torch
import torch.nn as nn

from ..core.loader.file import load_state_dict


class FourierEmbed(nn.Module):

    def __init__(self, num_freqs: int = 6):
        super().__init__()
        self.num_freqs = num_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frequencies = (
            2.0 ** torch.arange(self.num_freqs, device=x.device, dtype=x.dtype)
            * math.pi
        )
        embedded = x.unsqueeze(-1) * frequencies
        return torch.cat(
            [
                x,
                torch.sin(embedded).flatten(-2),
                torch.cos(embedded).flatten(-2),
            ],
            dim=-1,
        )


class CameraPoseEncoder(nn.Module):

    def __init__(self, dim: int, num_freqs: int = 6, hidden: int = 1024):
        super().__init__()
        self.fourier = FourierEmbed(num_freqs=num_freqs)
        input_dim = 12 * (1 + 2 * num_freqs)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    @staticmethod
    def _to_bf44(pose: torch.Tensor) -> torch.Tensor:
        if pose.ndim == 3:
            return pose.unsqueeze(0)
        if pose.ndim == 4:
            return pose
        raise ValueError(
            f"cam_pose must be (F,4,4) or (B,F,4,4), got {tuple(pose.shape)}"
        )

    def forward(self, cam_pose: torch.Tensor) -> torch.Tensor:
        cam_pose = self._to_bf44(cam_pose)
        pose = cam_pose[..., :3, :4].reshape(cam_pose.shape[0], cam_pose.shape[1], 12)
        return self.mlp(self.fourier(pose))


def load_camera_pose_encoder(
    pipe,
    checkpoint_path,
    device="cuda",
    dtype=torch.bfloat16,
    strict=True,
):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Camera-pose checkpoint not found: {checkpoint_path}")
    dim = pipe.dit.dim
    encoder = CameraPoseEncoder(dim=dim).to(device=device, dtype=dtype)
    prefixes = (
        "pipe.relpose_ecam.",
        "relpose_ecam.",
        "pipe.dit.relpose_ecam.",
        "dit.relpose_ecam.",
    )
    weights = {}
    for key, value in load_state_dict(str(checkpoint_path), device="cpu").items():
        for prefix in prefixes:
            if key.startswith(prefix):
                weights[key[len(prefix) :]] = value
                break
    if not weights:
        raise KeyError(f"No CameraPoseEncoder weights were found in {checkpoint_path}")
    output_weight = weights.get("mlp.4.weight")
    if output_weight is not None and output_weight.shape[0] != dim:
        raise ValueError(
            f"CameraPoseEncoder output dim {output_weight.shape[0]} does not match DiT dim {dim}"
        )
    encoder.load_state_dict(weights, strict=strict)
    encoder.eval()
    pipe.relpose_ecam = encoder
    pipe.dit.relpose_ecam = encoder
    if pipe.dit2 is not None:
        pipe.dit2.relpose_ecam = encoder
    return encoder
