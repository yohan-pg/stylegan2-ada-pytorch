from .prelude import *
from .variables import *


class Interpolator:
    def __init__(self, G):
        self.G = G

    def interpolate(
        self, A: Variable, B: Variable, num_interpolation_steps: int = 7
    ) -> List[Variable]:
        frames = []
        variables = []

        for i in range(num_interpolation_steps):
            alpha = i / (num_interpolation_steps - 1)
            variables.append(A.interpolate(B, alpha))
            frames.append((self.G.synthesis(variables[-1].to_styles()) + 1) / 2)

        return Interpolation(variables, frames)


@dataclass(eq=False)
class Interpolation:
    variables: List[Variable]
    frames: List[torch.Tensor]

    def grid(self):  # todo rename to these photo sequence things
        return torch.cat(self.frames)

    def save(self, out_path: str) -> None:
        # todo save an "interpolation" class instead
        save_image(self.grid(), out_path, nrow=len(self.frames[0]))

    def ppl(self, criterion) -> float:
        total = 0.0

        for a, b in zip(self.frames, self.frames[1:]):
            total += criterion(a, b).item()

        return total

    def endpoint_distance(self, criterion) -> float:
        return criterion(self.frames[0], self.frames[-1]).item()

    def latent_distance(self, criterion) -> float:
        return nn.MSELoss()(self.variables[0].params, self.variables[-1].params)
