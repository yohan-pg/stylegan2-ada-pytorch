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

    def from_data(self, data):
        return self.__class__(self.G[0], data)

    @staticmethod
    def from_variable(variable):
        return variable.__class__(variable.G[0], variable.data)

    @abstractclassmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        raise NotImplementedError

    @abstractclassmethod
    def sample_random_from(cls, G: nn.Module, batch_size: int = 1):
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

    def edit(self, data):
        return self.__class_(self.G[0], self.data + data)

    def direction_to(self, other):
        diff = other.data - self.data
        return self.from_data(diff / diff.norm(dim=-1, keepdim=True))

    def __add__(self, other: "Variable"):
        return self.from_data(self.data + other.data)

    def __sub__(self, other: "Variable"):
        return self.from_data(self.data - other.data)

    def __mul__(self, scalar: float):
        return self.from_data(self.data * scalar)

    def norm(self):
        return self.data.norm(dim=(1, 2))

    def to_image(self, const_noise: bool = True):
        return self.styles_to_image(self.to_styles(), const_noise)

    def disable_noise(self):
        return self

    def styles_to_image(self, styles, const_noise: bool = True):
        return (
            self.G[0].synthesis(styles, noise_mode="const" if const_noise else "random")
            + 1
        ) / 2

    def inform(exp_sq_avg):
        pass

    def split_into_individual_variables(self):
        return [
            self.__class__(
                self.G[0],
                nn.Parameter(p.unsqueeze(0))
                if isinstance(self.data, nn.Parameter)
                else p.unsqueeze(0),
            )
            for p in self.data
        ]

    @classmethod
    @torch.no_grad() 
    def find_init_point(cls, G, target, criterion, num_iters=10, batch_size=16):
        best_var = None
        best_distance = float("Inf")

        for i in tqdm.tqdm(range(num_iters)):
            var = cls.sample_random_from(G, batch_size)
            for sample, var in zip(
                var.to_image(), var.split_into_individual_variables()
            ):
                sample = sample.unsqueeze(0)
                if (distance := criterion(sample, target)).item() < best_distance:
                    best_distance = distance.item()
                    best_var = var
        
        return best_var
