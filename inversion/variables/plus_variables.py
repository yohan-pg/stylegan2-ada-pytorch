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
                super().to_styles(super().sample_from(G, batch_size=batch_size))
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


class WPlusVariable(_WPlusVariable, WVariable):
    pass


class _ZPlusVariable:
    space_name = "Z+"

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

class ZPlusVariable(_ZPlusVariable, ZVariable):
    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        return cls(
            G,
            nn.Parameter(
                torch.randn(batch_size, G.num_required_vectors() * G.num_ws, G.w_dim).cuda()
            ),
        )


class ZPlusVariableInitAtMean(_ZPlusVariable, ZVariableInitAtMean):
    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        return cls(
            G,
            nn.Parameter(
                torch.zeros(batch_size, G.num_required_vectors() * G.num_ws, G.w_dim).cuda()
            ),
        )