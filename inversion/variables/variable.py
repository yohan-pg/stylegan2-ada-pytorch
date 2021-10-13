from ..prelude import *


class Variable(ToStyles, ABC, nn.Module):
    @final
    def __init__(self, G, data: torch.Tensor):
        super().__init__()
        self.G = [G]
        if data.ndim == 3:
            pass
            # assert data.shape[1:] == (G.num_required_vectors(), G.w_dim)
        else:
            assert data.ndim == 2
            assert data.shape[1] == G.w_dim
        self.data = data
    
    @abstractclassmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        raise NotImplementedError

    @abstractmethod
    def interpolate(self, other: "Variable", alpha: float) -> "Variable":
        raise NotImplementedError

    def copy(self):
        data = self.data.clone()
        return self.__class__(
            self.G[0],
            nn.Parameter(data) if isinstance(self.data, nn.Parameter) else data,
        )

    def norm(self):
        return self.data.norm(dim=(1, 2))

    def to_image(self, const_noise: bool = True):
        return self.styles_to_image(self.to_styles(), const_noise)

    def styles_to_image(self, styles, const_noise: bool = True):
        return ( 
            self.G[0].synthesis(styles, noise_mode="const" if const_noise else "random")
            + 1
        ) / 2

    def split_into_individual_variables(self):
        return [self.__class__(self.G[0], p.unsqueeze(0)) for p in self.data]
