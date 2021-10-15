from .variable import *
from .w_variable import *


class ZVariable(Variable):
    space_name = "Zw"
    
    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        return cls(
            G,
            nn.Parameter(
                (
                    torch.randn(batch_size, G.num_required_vectors(), G.z_dim).cuda()
                ).squeeze(1)
            ),
        )

    def interpolate(self, other: "ZVariable", alpha: float) -> Variable:
        assert self.G == other.G

        def slerp(a, b, t):
            "StyleGAN's slerp code, slightly modified so that norms are endpoints preserved and linearly interpolated between at intermediate steps."
            na = a.norm(dim=-1, keepdim=True)
            nb = b.norm(dim=-1, keepdim=True)
            a = a / na
            b = b / nb
            d = (a * b).sum(dim=-1, keepdim=True)
            p = t * torch.acos(d)
            c = b - d * a
            c = c / c.norm(dim=-1, keepdim=True)
            d = a * torch.cos(p) + c * torch.sin(p)
            d = d / d.norm(dim=-1, keepdim=True)
            return d * ((1.0 - t) * na + t * nb)
        
        return self.__class__(
            self.G[0],
            slerp(
                self.data,
                other.data,
                alpha,
            ),
        )

    def to_W(self) -> WVariable:
        return WVariable(self.G[0], self.to_styles()[:, :self.G[0].num_required_vectors()])

    def interpolate_in_W(self, other: "ZVariable", alpha: float) -> Variable:
        return WVariable(
            self.G[0],
            lerp(
                self.to_styles()[:, : self.G[0].num_required_vectors()],
                other.to_styles()[:, : self.G[0].num_required_vectors()],
                alpha,
            ),
        )

    def to_styles(self):
        return self.G[0].mapping(self.data, None)


class ZVariableWithNoise(ZVariable):
    noise_gain = 0.1
    def to_styles(self):
        return self.G[0].mapping(self.data * torch.randn_like(self.data) * self.noise_gain, None)


class ZSkipVariable(ZVariable):
    interpolate = WVariable.interpolate 

    def to_styles(self):
        return self.G[0].mapping(self.data, None) +  WVariable.to_styles(self)