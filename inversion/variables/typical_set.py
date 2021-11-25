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


class ZVariableConvex(ZVariable):
    truncation_factor = 1.0

    N = 100

    samples = torch.randn(N, 512, 512).cuda()

    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1, init_scale=1.0):
        return cls(
            G,
            nn.Parameter(
                (nn.functional.softmax(torch.randn(batch_size, cls.N) / 100).cuda()).squeeze(1)
                * init_scale
            ),
        )

    def to_styles(self):
        print(nn.functional.softmax(self.data, dim=1))
        return self.G[0].mapping(
            (nn.functional.softmax(self.data, dim=1).unsqueeze(2).unsqueeze(3)
            * self.samples.unsqueeze(0)).sum(dim=1),
            None,
        )


class ZVariableConstrainToTypicalSetAllVecsDyn(ZVariable):
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
        norm = self.data.norm(dim=(1, 2), keepdim=True)
        target = (
            math.sqrt(self.data.shape[1])
            * math.sqrt(self.data.shape[2])
            * self.truncation_factor
        )

        return self.G[0].mapping(self.data / (norm + 1e-8) * target, None)


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


class ZVariableConstrainToTypicalSetAllVecsL1(ZVariable):
    truncation_factor = 1.0

    def after_step(self, diff):
        norm = self.data.norm(dim=(1, 2), keepdim=True, p=1)
        target = torch.ones(*self.data.shape[1:]).norm(p=1)

        with torch.no_grad():
            self.data.copy_(self.data / (norm + 1e-8) * target)


class ZVariableConstrainToTypicalSetAllVecsL3(ZVariable):
    truncation_factor = 1.0

    def after_step(self, diff):
        norm = self.data.norm(dim=(1, 2), keepdim=True, p=3)
        target = torch.ones(*self.data.shape[1:]).norm(p=3)

        with torch.no_grad():
            self.data.copy_(self.data / (norm + 1e-8) * target)


class ZVariableConstrainToTypicalSetAllVecsL0(ZVariable):
    truncation_factor = 1.0

    def after_step(self, diff):
        norm = self.data.norm(dim=(1, 2), keepdim=True, p=0)
        target = torch.ones(*self.data.shape[1:]).norm(p=0)

        with torch.no_grad():
            self.data.copy_(self.data / (norm + 1e-8) * target)


def ZVariableClamped(amount):
    class ZVariableClamped(ZVariableInitAtMean):
        def to_styles(self) -> Styles:
            return super().to_styles().clamp(min=-amount, max=amount)

    return ZVariableClamped
