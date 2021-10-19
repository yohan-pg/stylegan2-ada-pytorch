from .prelude import *
from .variables import *


@dataclass(eq=False)
class Interpolation:
    variables: List[Variable]
    frames: List[torch.Tensor]

    def grid(self):  # todo rename to these photo sequence things
        return torch.cat(self.frames)

    def save(self, out_path: str, target_A=None, target_B=None) -> None:
        grid = self.grid()
        if target_A is not None:
            grid = torch.cat([target_A] + [x.unsqueeze(0) for x in grid])
        if target_B is not None:
            grid = torch.cat([x.unsqueeze(0) for x in grid] + [target_B])
        batch_size = len(self.frames[0])
        save_image(grid, out_path, nrow=100 if batch_size == 1 else batch_size)

    def ppl(self, criterion) -> float:
        total = 0.0

        pairs = list(zip(self.frames, self.frames[1:]))
        assert len(pairs) == len(self.frames) - 1

        for a, b in pairs:
            dist = criterion(a, b).sqrt().item()
            assert dist >= 0
            total += dist

        assert total >= self.endpoint_distance(criterion), self.endpoint_distance(
            criterion
        )
        return total

    def endpoint_distance(self, criterion) -> float:
        return criterion(self.frames[0], self.frames[-1]).sqrt().item()

    def latent_distance(self, criterion) -> float:
        return criterion(self.variables[0].data, self.variables[-1].data)

    @staticmethod
    def from_variables(
        A: Variable, B: Variable, num_steps: int = 7, gain=1.0
    ) -> "Interpolation":
        frames = []
        variables = []

        for i in range(num_steps):
            alpha = i / (num_steps - 1) * gain
            variables.append(A.interpolate(B, alpha))
            frames.append(variables[-1].to_image())

        return Interpolation(variables, frames)

    @staticmethod
    def mix_between(
        A: Variable, B: Variable
    ):
        return Interpolation.from_variables(A, B, num_steps=3).frames[1]

