import torch


class FlowMatchScheduler:

    def __init__(self):
        self.num_train_timesteps = 1000
        self.training = False

    @staticmethod
    def set_timesteps_wan(num_inference_steps=100, denoising_strength=1.0, shift=None):
        shift = 5 if shift is None else shift
        sigma_start = denoising_strength
        sigmas = torch.linspace(sigma_start, 0.0, num_inference_steps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        return (sigmas, sigmas * 1000)

    def set_training_weight(self):
        steps = 1000
        values = torch.exp(-2 * ((self.timesteps - steps / 2) / steps) ** 2)
        values = values - values.min()
        weights = values * (steps / values.sum())
        if len(self.timesteps) != steps:
            weights = weights * (len(self.timesteps) / steps) + weights[1]
        self.linear_timesteps_weights = weights

    def set_timesteps(
        self, num_inference_steps=100, denoising_strength=1.0, training=False, **kwargs
    ):
        self.sigmas, self.timesteps = self.set_timesteps_wan(
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
            **kwargs
        )
        self.training = training
        if training:
            self.set_training_weight()

    def step(self, model_output, timestep, sample):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sigma_next = (
            0
            if timestep_id + 1 >= len(self.timesteps)
            else self.sigmas[timestep_id + 1]
        )
        return sample + model_output * (sigma_next - sigma)

    def add_noise(self, original_samples, noise, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        return (1 - sigma) * original_samples + sigma * noise

    def training_target(self, sample, noise, timestep):
        return noise - sample

    def training_weight(self, timestep):
        timestep_id = torch.argmin(
            (self.timesteps - timestep.to(self.timesteps.device)).abs()
        )
        return self.linear_timesteps_weights[timestep_id]
