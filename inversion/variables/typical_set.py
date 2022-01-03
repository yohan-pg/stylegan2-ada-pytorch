from .w_variable import *
from .z_variable import *

from .init_at_mean import *


class ZVariableConstrainToTypicalSet(ZVariable):
    truncation_factor = 1.0

    def to_styles(self) -> Styles:
        with torch.no_grad():
            self.data.copy_(normalize_2nd_moment(self.data) * self.truncation_factor)
        return super().to_styles()


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

    def interpolate(
        self, other: "ZVariableClampToTypicalSet", alpha: float
    ) -> Variable:
        assert ZVariable.from_variable(self).interpolate(
            ZVariable.from_variable(other), alpha
        )


class ZVariableConstrainToTypicalSetAllVecs(ZVariable):
    truncation_factor = 1.0

    def after_step(self):
        norm = self.data.norm(dim=(1, 2), keepdim=True)
        target = (
            math.sqrt(self.data.shape[1])
            * math.sqrt(self.data.shape[2])
            * self.truncation_factor
        )

        with torch.no_grad():
            self.data.copy_(self.data / (norm + 1e-8) * target)

    def to_styles(self):
        return super().to_styles()

class ZVariableConstrainToTypicalSetAllVecsInW(ZVariable):
    truncation_factor = 1.0

    def after_step(self):
        norm = self.data.norm(dim=(1, 2), keepdim=True)
        target = (
            math.sqrt(self.data.shape[1])
            * math.sqrt(self.data.shape[2])
            * self.truncation_factor
        )

        with torch.no_grad():
            self.data.copy_(self.data / (norm + 1e-8) * target)

    def to_styles(self):
        styles = super().to_styles()
        norm = styles.norm(dim=(1, 2), keepdim=True)
        target = (
            math.sqrt(self.data.shape[1])
            * math.sqrt(self.data.shape[2])
            * self.truncation_factor
        )
        return styles / (norm + 1e-8) * target


class ZVariableConstrainToTypicalSetInW(ZVariable):
    truncation_factor = 1.0

    def after_step(self):
        norm = self.data.norm(dim=(1, 2), keepdim=True)
        target = (
            math.sqrt(self.data.shape[1])
            * math.sqrt(self.data.shape[2])
            * self.truncation_factor
        )

        with torch.no_grad():
            self.data.copy_(self.data / (norm + 1e-8) * target)

    def to_styles(self):
        styles = super().to_styles()
        norm = styles.norm(dim=(2), keepdim=True)
        target = math.sqrt(self.data.shape[2]) * self.truncation_factor
        return styles / (norm + 1e-8) * target


class ZVariableConstrainToTypicalSetAllVecsInWWithNoise(ZVariable):
    truncation_factor = 1.0
    noise_level = 0.2

    def after_step(self, diff):
        with torch.no_grad():
            self.data += self.noise_level * torch.randn_like(self.data)
        norm = self.data.norm(dim=(1, 2), keepdim=True)
        target = (
            math.sqrt(self.data.shape[1])
            * math.sqrt(self.data.shape[2])
            * self.truncation_factor
        )

        with torch.no_grad():
            self.data.copy_(self.data / (norm + 1e-8) * target)

    def to_styles(self):
        styles = super().to_styles()
        norm = styles.norm(dim=(1, 2), keepdim=True)
        target = (
            math.sqrt(self.data.shape[1])
            * math.sqrt(self.data.shape[2])
            * self.truncation_factor
        )
        return styles / (norm + 1e-8) * target
