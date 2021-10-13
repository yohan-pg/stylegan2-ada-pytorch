from .variable import *


def lerp(a, b, alpha):
    return (1.0 - alpha) * a + alpha * b


class WVariable(Variable):
    init_at_mean = True

    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        if cls.init_at_mean:
            data = G.mapping(
                (
                    torch.zeros(batch_size, G.num_required_vectors(), G.z_dim)
                    .squeeze(1)
                    .cuda()
                ),
                None,
                skip_w_avg_update=True,
            )[:, : G.num_required_vectors(), :]
        else:
            data = G.mapping(
                (
                    torch.randn(batch_size, G.num_required_vectors(), G.z_dim)
                    .squeeze(1)
                    .cuda()
                ),
                None,
                skip_w_avg_update=True,
            )[:, : G.num_required_vectors(), :]

        return cls(
            G,
            nn.Parameter(data),
        )

    def interpolate(self, other: "WVariable", alpha: float) -> Styles:
        assert self.G == other.G
        return self.__class__(self.G[0], lerp(self.data, other.data, alpha))

    def to_styles(self) -> Styles:
        data = self.data
        
        if self.init_at_mean:
            G = self.G[0]
            mean = G.mapping.w_avg.reshape(1, 1, G.w_dim).repeat(
                len(data), G.num_required_vectors(), 1
            )
            data = (data + mean)
        
        return data.repeat(1, self.G[0].num_ws, 1)


class WVariableWithNoise(WVariable):
    noise_gain = 0.1

    def to_styles(self) -> Styles:
        return (self.data + torch.randn_like(self.data) * self.noise_gain).repeat(1, self.G[0].num_ws, 1)



class WVariableEarlyLayers(WVariable):
    num_layers = 10

    def to_styles(self) -> Styles:
        styles = super().to_styles()
        mean = WVariable.sample_from(self.G[0], len(self.data)).data.repeat(1, self.G[0].num_ws, 1)
        
        return torch.cat((
            styles[:, :512 * self.num_layers],
            mean[:, 512 * self.num_layers:]
        ), dim=1)


class WVariableLastLayers(WVariable):
    num_layers = 8

    def to_styles(self) -> Styles:
        styles = super().to_styles()
        mean = WVariable.sample_from(self.G[0], len(self.data)).data.repeat(1, self.G[0].num_ws, 1)
        
        return torch.cat((
            mean[:, :512 * self.num_layers],
            styles[:, 512 * self.num_layers:]
        ), dim=1)
