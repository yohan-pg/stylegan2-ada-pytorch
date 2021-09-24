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

    def compute_ppl(self, criterion) -> torch.Tensor:
        total = 0.0

        for a, b in zip(self.frames, self.preds[1:]):
            total += criterion(a, b).item() / len(self.preds)

        return total
