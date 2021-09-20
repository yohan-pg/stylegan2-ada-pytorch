from .criterions import *
from .variables import *
from .jittering import *
from .optimizer import *


class Inverter:
    def __init__(self, G, learning_rate: float = 0.1):
        self.G = copy.deepcopy(G).eval().requires_grad_(False)

    def create_optimizer(self, variable: Variable):
        return torch.optim.Adam(
            variable.parameters(),
            betas=(0.9, 0.999),
            lr=self.learning_rae,
        )

    def invert(
        self,
        # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        target: torch.Tensor,
        variable: Variable,
        criterion: InversionCriterion,
        num_steps: int,
    ):
        assert target.shape == (
            self.G.img_channels,
            self.G.img_resolution,
            self.G.img_resolution,
        )

        optimizer = self.create_optimizer(variable)

        for step in range(self.num_iterations):
            t = step / num_steps

            pred = self.G.synthesis(self.variable_to_style(), noise_mode="const")
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            yield loss, pred

