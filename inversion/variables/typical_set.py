from .w_variable import *
from .z_variable import *

from .init_at_mean import *


class ZVariableConstrainToTypicalSet(ZVariable):
    truncation_factor = 1.0

    def to_styles(self) -> Styles:
        with torch.no_grad():
            self.data.copy_(normalize_2nd_moment(self.data) * self.truncation_factor)
        return super().to_styles(self)


class ZVariableClampToTypicalSet(ZVariableInitAtMean):
    truncation_factor = 1.0

    def to_styles(self) -> Styles:
        with torch.no_grad():
            normalized = normalize_2nd_moment(self.data) * self.truncation_factor
            self.data.copy_(
                torch.where(
                    normalized.norm(dim=2, keepdim=True)
                    > self.data.norm(dim=2, keepdim=True),
                    self.data,
                    normalized,
                )
            )

        return super().to_styles()


class ZVariableClampToTypicalSetAllVecs(ZVariable):
    truncation_factor = 1.0

    def to_styles(self) -> Styles:
        norm = self.data.norm(dim=(1, 2), keepdim=True)
        target = math.sqrt(self.data.shape[1]) * math.sqrt(self.data.shape[2]) * self.truncation_factor

        with torch.no_grad():
            self.data.copy_(
                torch.where(
                    norm > target,
                    self.data / norm * target,
                    self.data,
                )
            )

        return super().to_styles()
