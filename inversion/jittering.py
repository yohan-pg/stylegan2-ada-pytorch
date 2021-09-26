from .prelude import *

class OptimizationConstraint:
    pass


class StyleJittering(OptimizationConstraint):
    def __init__(self, variable, initial_noise_factor=0.05, noise_ramp_length=0.75):
        super().__init__()
        self.variable = variable
        self.initial_noise_factor = initial_noise_factor
        self.noise_ramp_length = noise_ramp_length

    def update(self, t):
        w = self.variable.variable_to_style() # todo is this really applied after style mixing?
        self.w_noise_scale = (
            self.w_std
            * self.initial_noise_factor
            * max(0.0, 1.0 - t / self.noise_ramp_length) ** 2
        )
        w_noise = torch.randn_like(w) * w_noise_scale

    def to_styles(self):
        # todo, should this be before replication?
        # return w + w_noise # !!
        return self.variable.to_styles() 



class NoiseJittering(OptimizationConstraint):
    # initial_noise_factor=0.05,
    # noise_ramp_length=0.75,
    # regularize_noise_weight=1e5,
    # todo this must be in optimizer
    # + (list(noise_bufs.values()) if optimize_noise else []),

    def __init__(self, G):
        self.noise_bufs = {
            name: buf
            for (name, buf) in G.synthesis.named_buffers()
            if "noise_const" in name
        }

        for buf in self.noise_bufs.values():
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

    def update(self, t):
        # Noise regularization.
        reg_loss = 0.0
        for v in self.noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        # loss = dist + reg_loss * regularize_noise_weight

        # # Normalize noise.
        # with torch.no_grad():
        #     for buf in noise_bufs.values():
        #         buf -= buf.mean()
        #         buf *= buf.square().mean().rsqrt()