from .variable import *
from .w_variable import *


class GVariable(Variable):
    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        return cls(
            G,
            list(G.parameters()),
        )

    def interpolate(self, other: "GVariable", alpha: float) -> Variable:
        raise NotImplementedError

    def to_styles(self):
        return self.G[0].mapping(self.data, None)

