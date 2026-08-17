import torch


def parse_device_type(device):
    if isinstance(device, torch.device):
        return device.type
    if str(device).startswith("cuda"):
        return "cuda"
    if str(device).startswith("npu"):
        return "npu"
    return "cpu"
