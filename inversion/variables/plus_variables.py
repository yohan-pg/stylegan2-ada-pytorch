from .w_variable import *
from .z_variable import *

from .init_at_mean import *


class _WPlusVariable(ABC):
    space_name = "W+"

    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        return cls(
            G,
            nn.Parameter(
                WVariable.to_styles(WVariable.sample_from(G, batch_size=batch_size))
            ),
        )

    @classmethod
    def sample_random_from(cls, G: nn.Module, batch_size: int = 1):
        data = G.mapping(
            (
                torch.randn(batch_size, G.num_required_vectors(), G.z_dim)
                .squeeze(1)
                .cuda()
            ),
            None,
            skip_w_avg_update=True,
        )

        return cls(
            G,
            nn.Parameter(data),
        )

    def to_styles(self) -> Styles:
        return self.data

    def mix(self, other: "WPlusVariable", num_layers: float):
        split_point = num_layers * self.G[0].num_required_vectors()

        return WPlusVariable(
            self.G[0],
            torch.cat(
                (self.data[:, :split_point, :], other.data[:, split_point:, :]), dim=1
            ),
        )


def to_W_plus(self):
    return WPlusVariable(self.G[0], self.to_styles())


WVariable.to_W_plus = to_W_plus


class WPlusVariable(_WPlusVariable, WVariable):
    pass


class WPlusVariableMultiSpeed(WPlusVariable):
    speeds = torch.ones(16)

    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        wp = WPlusVariable.sample_from(G, batch_size) 

        assert len(cls.speeds) == G.mapping.num_ws

        cls.speeds = cls.speeds.to(wp.data.device)
        
        with torch.no_grad():
            wp.data /= cls._speeds(G)
        
        return cls(
            G,
            wp.data
        )
    
    @classmethod
    def _speeds(cls, G):
        return cls.speeds.unsqueeze(1).repeat_interleave(G.num_required_vectors(), 0)

    def to_styles(self):
        return self.data * self._speeds(self.G[0])

class WPlusVariableInterpReg(WPlusVariable):
    alpha = 0.9

    def to_styles(self):
        return self.interpolate(WPlusVariable.sample_random_from(self.G[0], len(self.data)), 2.0).data


class WPlusVariableClamped(_WPlusVariable, WVariable):
    clamp_size = 0.1

    def to_styles(self) -> Styles:
        return self.data.clamp(min=-self.clamp_size, max=self.clamp_size)


class WPlusVariableMixingReg(WPlusVariable):
    prob = 0.1

    def to_styles(self):
        data = self.data.clone()

        if self.training:
            selection = torch.rand(data.size(1)) < self.prob
            data[:, selection] = self.G[0].mapping(
                torch.randn((1, self.G[0].num_required_vectors(), data.size(2))).to(
                    data.device
                ),
                None,
            )[:, selection]

        return data

    def to_W(self):
        return self


class WPlusVariableRandomInit(WPlusVariable):
    init_truncation = 0.25

    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        return cls(
            G,
            nn.Parameter(
                WVariable.sample_random_from(
                    G, batch_size, truncation_psi=cls.init_truncation
                ).to_styles()
            ),
        )


class _ZPlusVariable:
    space_name = "Z+"

    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        return cls(
            G,
            nn.Parameter(
                torch.zeros(
                    batch_size, G.num_required_vectors() * G.num_ws, G.w_dim
                ).cuda()
            ),
        )

    def to_styles(self, data=None):
        if data is None:
            data = self.data
        n = self.G[0].num_required_vectors()
        data = data + torch.randn_like(data) * self.noise_amount
        return torch.cat(
            [
                self.G[0].mapping(data[:, i * n : (i + 1) * n].squeeze(1), None)[
                    :, :n
                ]
                for i in range(self.G[0].num_ws)
            ],
            dim=1,
        )


class ZPlusVariable(_ZPlusVariable, ZVariable):
    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        return cls(
            G,
            nn.Parameter(
                torch.randn(
                    batch_size, G.num_required_vectors() * G.num_ws, G.w_dim
                ).cuda()
            ),
        )

    def restrict(self, psi):
        
        pass


class ZPlusVariableConstrainToTypicalSetAllVecs(ZPlusVariable):
    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        return cls(
            G,
            nn.Parameter(
                torch.randn(batch_size, G.num_required_vectors(), G.w_dim)
                .repeat(1, G.num_ws, 1)
                .cuda()
            ),
        )

    def after_step(self):
        delta = self.data
        norm = delta.norm(dim=(1, 2), keepdim=True)
        target = math.sqrt(self.data.shape[1]) * math.sqrt(self.data.shape[2])

        with torch.no_grad():
            self.data.copy_(delta / (norm + 1e-8) * target)


def make_ZVariablePlusWithFeatureAlphaDropoutOnZ(p, g=1.0):
    class ZVariablePlusWithFeatureAlphaDropoutOnZ(ZPlusVariable):
        prob = p
        dropout = nn.FeatureAlphaDropout(prob)
        truncation_factor = 1.0
        gamma = g

        num_steps = 0

        def after_step(self):
            self.dropout.p *= self.gamma

        def to_styles(self):
            return ZPlusVariable.to_styles(self, self.dropout(self.data))

    return ZVariablePlusWithFeatureAlphaDropoutOnZ


def make_ZVariableMixingReg(p, g=1.0):
    class ZVariableMixingReg(ZVariable):
        prob = p
        truncation_factor = 1.0
        gamma = g
        num_steps = 0

        def after_step(self):
            self.prob *= self.gamma

        def to_styles(self):
            data = self.data.clone()
            if self.training:
                selection = torch.rand(data.size(1)) < self.prob
                data[:, selection] = torch.randn_like(data)[:, selection]
            return ZVariable.to_styles(self, data)

        def to_W(self):
            return self

    return ZVariableMixingReg


def make_ZVariablePlusWithMixingReg(p, g=1.0):
    class ZVariablePlusWithMixingReg(ZPlusVariable):
        prob = p
        truncation_factor = 1.0
        gamma = g
        num_steps = 0

        def after_step(self):
            self.prob *= self.gamma

        def to_styles(self):
            n = self.G[0].num_required_vectors()
            data = self.data.clone()
            if self.training:
                selection = torch.rand(data.size(1)) < self.prob
                data[:, selection] = torch.randn_like(data)[:, selection]
            return torch.cat(
                [
                    self.G[0].mapping(data[:, i * n : (i + 1) * n].squeeze(1), None)[
                        :, :n
                    ]
                    for i in range(self.G[0].num_ws)
                ],
                dim=1,
            )

        def to_W(self):
            return self

    return ZVariablePlusWithMixingReg


class ZPlusVariableInitAtMean(_ZPlusVariable, ZVariableInitAtMean):
    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        return cls(
            G,
            nn.Parameter(
                torch.zeros(
                    batch_size, G.num_required_vectors() * G.num_ws, G.w_dim
                ).cuda()
            ),
        )
