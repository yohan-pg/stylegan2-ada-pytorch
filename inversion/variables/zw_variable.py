from .variable import *

from .w_variable import *
from .z_variable import *

class CombinedZWVariable(Variable):
    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        return cls(
            G,
            nn.ParameterList(
                WVariable.sample_from(G, batch_size).data,
                ZVariable.sample_from(G, batch_size).data,
            ),
        )

    def interpolate(self, other: "WVariable", alpha: float) -> Styles:
        assert self.G == other.G
        return self.__class__(
            self.G[0], lerp(self.data, other.data, alpha)
        )

    def to_styles(self) -> Styles:
        return self.data.repeat(1, self.G[0].num_ws, 1)
