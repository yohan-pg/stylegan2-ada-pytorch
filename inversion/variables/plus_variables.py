from .w_variable import *
from .z_variable import *

from .init_at_mean import *


class _WPlusVariable(ABC):
    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        return cls(
            G,
            nn.Parameter(
                super().to_styles(super().sample_from(G, batch_size=batch_size))
            ),
        )

    def to_styles(self) -> Styles:
        return self.data


class WPlusVariable(_WPlusVariable, WVariable):
    pass


# class ZPlusVariable(PlusVariable, ZVariable):
#     pass


class ZPlusVariableInitAtMean(ZVariableInitAtMean):
    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        return cls(
            G,
            nn.Parameter(
                torch.zeros(batch_size, G.num_required_vectors() * G.num_ws, G.w_dim).cuda()
            ),
        )

    def to_styles(self):
        n = self.G[0].num_required_vectors()
        return torch.cat(
            [
                self.G[0].mapping(self.data[:, i * n : (i + 1) * n].squeeze(1), None)[:, :n]
                for i in range(self.G[0].num_ws)
            ],
            dim=1
        )