import torch


def FlowMatchSFTLoss(pipe, **inputs):
    max_timestep = int(
        inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps)
    )
    min_timestep = int(
        inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps)
    )
    timestep_id = torch.randint(min_timestep, max_timestep, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(
        dtype=pipe.torch_dtype, device=pipe.device
    )
    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(
        inputs["input_latents"], noise, timestep
    )
    target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    prediction = pipe.model_fn(**models, **inputs, timestep=timestep)
    loss = torch.nn.functional.mse_loss(prediction.float(), target.float())
    return loss * pipe.scheduler.training_weight(timestep)
