from .variable import *
from .w_variable import *


class GVariable(Variable):
    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        G.train
        return cls(
            G,
            torch.nn.ParameterList(
                [nn.Parameter(WVariable.sample_random_from(G, batch_size).to_styles())]
                # + [nn.Parameter(torch.zeros_like(param)) for param in G.parameters()]
                + list(G.parameters())
            ),
        )

    @classmethod
    def sample_random_from(cls, G: nn.Module, batch_size: int = 1):
        raise NotImplementedError

    def interpolate(self, other: "GVariable", alpha: float) -> Variable:
        return self.__class__(
            self.G[0],
            nn.ParameterList(
                [
                    nn.Parameter(a.lerp(b, alpha))
                    for a, b in zip(self.data, other.data)
                ]
            ),
        )

    def to_image(self):
        backups = {}

        # with torch.no_grad():
        #     for param, offset in zip(self.G[0].parameters(), self.data[1:]):
        #         param.add_(offset)
        #         backups[param] = param.clone()
        
        # for a, b in zip(self.G[0].parameters(), self.data[1:]):
        #     a.copy_(b)

        image = self.G[0].synthesis(self.data[0])

        # with torch.no_grad():
        #     for param, offset in zip(self.G[0].parameters(), self.data[1:]):
        #         param.copy_(backups[param])

        return image

    def to_styles(self):
        raise NotImplementedError
