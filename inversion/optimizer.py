from .prelude import *


@dataclass(eq=False)
class RampedDownAdam:
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
