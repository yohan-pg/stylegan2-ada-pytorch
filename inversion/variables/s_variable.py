from .variable import *
from .w_variable import *


class SVariable(Variable):
    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        raise NotImplementedError

    def interpolate(self, other: "SVariable", alpha: float) -> Variable:
        raise NotImplementedError

    def to_styles(self):
        raise NotImplementedError

