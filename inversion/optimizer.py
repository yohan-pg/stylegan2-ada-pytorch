from .prelude import *
import torch.optim.lr_scheduler as lr_scheduler

import torch.optim


class HackerBlend(torch.optim.Optimizer):
    def __init__(self, params, sgd_lr, adam_lr, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, dict(sgd_lr=sgd_lr, adam_lr=adam_lr, eps=eps, betas=betas))
        self.sgd = torch.optim.SGD(lr=sgd_lr)
        self.adam = torch.optim.Adam(lr=adam_lr)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            print(lr)
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is not None:
                    p -= lr * p.grad.sign()

        return loss

class SignGD(torch.optim.Optimizer):
    def __init__(self, params, lr, eps=0.0):
        super().__init__(params, dict(lr=lr, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            print(lr)
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is not None:
                    p -= lr * p.grad.sign()

        return loss


class SGDMisconditionned(torch.optim.Optimizer):
    def __init__(self, params, lr, condition=None):
        super().__init__(params, dict(lr=lr, condition=condition))
        self.condition = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]

            if group["condition"] is None:
                group["condition"] = [
                    torch.rand_like(p)
                    for p in group["params"]
                ]

            for p, c in zip(group["params"], group["condition"]):
                if p.grad is not None:
                    p -= lr * p.grad / (c + 1e-8)

        return loss


class SGDGamma(torch.optim.Optimizer):
    def __init__(self, params, lr, gamma=0.0, scale=1.0):
        super().__init__(params, dict(lr=lr, gamma=gamma, scale=scale))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            gamma = group["gamma"]
            scale = group["scale"]

            for p in group["params"]:
                if p.grad is not None:
                    p -= lr * p.grad.sign() * (scale * p.grad)**gamma

        return loss


class SGDClamped(torch.optim.Optimizer):
    def __init__(self, params, lr, max=float("Inf"), min=0.0):
        super().__init__(params, dict(lr=lr, max=max, min=min))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            max = group["max"]
            min = group["min"]

            for p in group["params"]:
                if p.grad is not None:
                    p -= lr * p.grad.sign() * p.grad.abs().clamp(min=min, max=max)

        return loss


class SignGDOverQuantile(torch.optim.Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, dict(lr=lr))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is not None:
                    mask = (
                            p.grad.abs()
                            > p.grad.abs()
                            .reshape(p.size(0), -1)
                            .quantile(0.999, dim=1)
                            .unsqueeze(1)
                            .unsqueeze(2)
                        )
                    print( mask.sum() / mask.numel() )
                    p -= (
                        lr
                        * p.grad.sign()
                        * mask
                    )

        return loss


class SignGDClamped(torch.optim.Optimizer):
    def __init__(self, params, lr, min=0.0):
        super().__init__(params, dict(lr=lr, min=min))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            min = group["min"]

            for p in group["params"]:
                if p.grad is not None:
                    p -= lr * p.grad.sign() * (p.grad.abs() > min)

        return loss


class SignGDMix(torch.optim.Optimizer):
    def __init__(self, params, lr, alpha=100.0):
        super().__init__(params, dict(lr=lr, alpha=alpha))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]

            for p in group["params"]:
                if p.grad is not None:
                    p -= lr * (p.grad.sign() + alpha * p.grad)

        return loss


@dataclass(eq=False)
class StyleGAN2Adam:
    initial_learning_rate: float = 0.1
    lr_rampdown_length: float = 0.25
    lr_rampup_length: float = 0.05

    def update(self, t):
        lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
        lr = self.initial_learning_rate * lr_ramp

        # for param_group in optimizer.param_groups:
        #     param_group["lr"] = lr


class _OptimizerWithNoise:
    def __init__(
        self, parameters, noise_amount: float = 0.0, noise_gamma=1.0, noise_sparsness=0.0, **kwargs
    ):
        super().__init__(parameters, **kwargs)
        self.noise_amount = noise_amount
        self.current_noise_amount = noise_amount
        self.noise_gamma = noise_gamma
        self.noise_sparsness = noise_sparsness

    def step(self, *args, **kwargs):
        super().step(*args, **kwargs)
        self.current_noise_amount *= self.noise_gamma
        with torch.no_grad():
            for param_group in self.param_groups:
                for param in param_group["params"]:
                    mask = (torch.rand_like(param) > self.noise_sparsness)
                    param += (
                        self.current_noise_amount * mask
                        * math.sqrt(2.0 * param_group["lr"])
                        * torch.randn_like(param)
                    )


class AdamWithNoiseNaive(_OptimizerWithNoise, torch.optim.Adam):
    pass


class AdamWithNoise(_OptimizerWithNoise, torch.optim.Adam):
    def step(self, *args, **kwargs):
        torch.optim.Adam.step(self, *args, **kwargs)
        self.current_noise_amount *= self.noise_gamma
        with torch.no_grad():
            for param_group in self.param_groups:
                for param in param_group["params"]:
                    mask = (torch.rand_like(param) > self.noise_sparsness)
                    param += (
                        self.current_noise_amount * mask
                        * math.sqrt(2.0 * param_group["lr"])
                        * torch.randn_like(param)
                        / (
                            self.state[param]["exp_avg_sq"].sqrt() + param_group["eps"]
                        ).sqrt()
                    )



class AdamWithDropout(_OptimizerWithNoise, torch.optim.Adam):
    def __init__(self, dropout_prob):
        self.dropout_prob = dropout_prob
        self.backups = {}
        self.old_masks = {}

    @torch.no_grad()
    def step(self, *args, **kwargs):
        for param_group in self.param_groups:
            for param in param_group["params"]:
                mask = (torch.rand_like(param) < self.dropout_prob)
                param.copy_(self.backups[param])
                self.backups[param] = param.detach().clone()
                param += (
                    self.current_noise_amount * mask
                    * math.sqrt(2.0 * param_group["lr"])
                    * torch.randn_like(param)
                    / (
                        self.state[param]["exp_avg_sq"].sqrt() + param_group["eps"]
                    ).sqrt()
                )



class MixedOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizer_list):
        self.optimizer_list = optimizer_list
        self.num_steps = 0
        self.param_groups = []

        for _, optimizer in self.optimizer_list:
            self.param_groups += optimizer.param_groups

    @torch.no_grad()
    def step(self, closure=None):
        self.num_steps += 1
        max_steps, _ = self.optimizer_list[0]
        if max_steps <= self.num_steps:
            self.optimizer_list.pop(0)
        _, optimizer = self.optimizer_list[0]
        optimizer.step(closure)

class SignGDWithNoise(_OptimizerWithNoise, SignGD):
    pass


class SGDWithNoise(_OptimizerWithNoise, torch.optim.SGD):
    pass


class _OptimizerWithGreedyNoise(_OptimizerWithNoise):
    def __init__(
        self,
        parameters,
        *args,
        **kwargs,
    ):
        super().__init__(parameters, *args, **kwargs)
        assert len(parameters) == 1
        self.sample = parameters[0]
        self.sample_loss = None

    def step(self, compute_loss):
        with torch.no_grad():
            self.sample_loss = compute_loss().item()

        while True:
            backup = self.sample.detach().clone()
            self.zero_grad()
            super().step(compute_loss)
            with torch.no_grad():
                loss = compute_loss().item()
            if loss < self.sample_loss:
                print("Accept")
                return
            else:
                print("Reject")
                with torch.no_grad():
                    self.sample.copy_(backup)


class SGDWithNGreedyNoise(_OptimizerWithGreedyNoise, torch.optim.SGD):
    pass


class _MALAOptimizerMixin:
    # todo have the noise amount scale with a sqrt and everything
    def __init__(
        self,
        parameters,
        noise_amount: float = 1.0,
        sharpness: float = 1.0,
        max_attempts: int = 1000,
        *args,
        **kwargs,
    ):
        super().__init__(parameters, *args, **kwargs)
        self.noise_amount = noise_amount
        self.sharpness = sharpness
        self.max_attempts = max_attempts
        assert len(parameters) == 1
        self.sample = parameters[0]
        self.sample_loss = None

    def add_noise(self):
        lr = self.param_groups[0]["lr"]
        with torch.no_grad():
            self.sample.add_( #! temp removed lr
                math.sqrt(2.0 * self.noise_amount) * torch.randn_like(self.sample)
            )

    def take_step(self, compute_loss):
        self.zero_grad()
        super().step(compute_loss)

    #! compute_loss overwrites the loss in invertert

    def sample_proposal(self, compute_loss):
        backup = self.sample.detach().clone()

        self.take_step(compute_loss)
        proposal_mean = self.sample.clone()
        self.add_noise()
        proposal = self.sample.clone()
        with torch.no_grad():
            proposal_loss = compute_loss()

        self.take_step(compute_loss)
        next_proposal_mean = self.sample.clone()

        with torch.no_grad():
            self.sample.copy_(backup)
        return proposal_mean, proposal, proposal_loss, next_proposal_mean

    def link_logit(self, loss):
        return -self.sharpness * loss

    def transition_density_logit(self, from_point, to_point):
        lr = self.param_groups[0]["lr"]
        return -(1 / 4) * lr * torch.norm(to_point - from_point) #!!! dim is missing

    def accept(self, proposal_mean, proposal, proposal_loss, next_proposal_mean):
        print("wtf!", self.transition_density_logit(self.sample, next_proposal_mean))
        acceptance_rate = min(
            1.0,
            torch.exp(
                (
                    #! double check args
                    self.link_logit(proposal_loss)
                    + self.transition_density_logit(self.sample, next_proposal_mean)
                )
                - (
                    self.link_logit(self.sample_loss)
                    + self.transition_density_logit(proposal, proposal_mean)
                )
            ).item(),
        )
        return torch.rand(1).item() < acceptance_rate

    def step(self, compute_loss):
        if self.sample_loss is None:
            with torch.no_grad():
                self.sample_loss = compute_loss().item()

        for _ in range(self.max_attempts):  #!! this infinite loops!WTH!
            (
                proposal_mean,
                proposal,
                proposal_loss,
                next_proposal_mean,
            ) = self.sample_proposal(compute_loss)

            with torch.no_grad():
                if self.accept(
                    proposal_mean, proposal, proposal_loss, next_proposal_mean
                ):
                    print("Accept")
                    self.sample.copy_(proposal)
                    self.sample_loss = proposal_loss.item()
                    return
                else:
                    print("Reject")

        raise Exception(
            f"Failed to find an accepted sample over {self.max_attempts} attempts; rejection rate is too high."
        )


class MALASGD(_MALAOptimizerMixin, torch.optim.SGD):
    pass


class MALASignGD(_MALAOptimizerMixin, SignGD):
    pass


class MALAAdamNaive(_MALAOptimizerMixin, torch.optim.Adam):
    pass





class MetropolisOptimizier(torch.optim.Optimizer):
    # todo have the noise amount scale with a sqrt and everything
    def __init__(
        self,
        parameters,
        lr=1.0,
        sharpness: float = 1.0,
        max_attempts: int = 1000,
    ):
        super().__init__(parameters, dict(lr=lr))
        self.sharpness = sharpness
        self.max_attempts = max_attempts
        assert len(parameters) == 1
        self.sample = parameters[0]
        self.sample_loss = None

    def add_noise(self):
        lr = self.param_groups[0]["lr"]
        with torch.no_grad():
            self.sample.add_(
                math.sqrt(2.0 * lr) * torch.randn_like(self.sample)
            ) #! sqrt is weird but makes it easier to compare to langevin

    def link_logit(self, loss):
        return -self.sharpness * loss

    def accept(self, proposal_loss):
        acceptance_rate = min(
            1.0,
            torch.exp(
                (
                    self.link_logit(proposal_loss)
                )
                - (
                    self.link_logit(self.sample_loss)
                )
            ).item(),
        )
        return torch.rand(1).item() < acceptance_rate

    def sample_proposal(self, compute_loss):
        backup = self.sample.detach().clone()
        self.add_noise()
        proposal = self.sample.detach().clone()

        with torch.no_grad():
            proposal_loss = compute_loss() 
            self.sample.copy_(backup)
        
        return proposal, proposal_loss

    def step(self, compute_loss):
        if self.sample_loss is None:
            with torch.no_grad():
                self.sample_loss = compute_loss().item()

        for _ in range(self.max_attempts):
            (
                proposal,
                proposal_loss,
            ) = self.sample_proposal(compute_loss)

            with torch.no_grad():
                if self.accept(
                    proposal_loss
                ):
                    print("Accept")
                    self.sample.copy_(proposal)
                    self.sample_loss = proposal_loss.item()
                    return
                else:
                    print(".")

        raise Exception(
            f"Failed to find an accepted sample over {self.max_attempts} attempts; rejection rate is too high."
        )