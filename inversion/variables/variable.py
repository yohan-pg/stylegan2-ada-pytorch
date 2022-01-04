from ..prelude import *


class ParametrizableClass:
    @classmethod
    def parameterize(cls, **kwargs):
        class Wrapper(cls):
            pass

        for name, value in kwargs.items():
            setattr(Wrapper, name, value)
        Wrapper.__name__ = cls.__name__
        Wrapper.__qualname__ = cls.__qualname__
        return Wrapper


class Variable(ToStyles, ABC, nn.Module, ParametrizableClass):
    def __init__(self, G, data: torch.Tensor):
        super().__init__()
        self.G = [G]
        self.data = data

    def from_data(self, data):
        return self.__class__(self.G[0], data)

    def detach(self):
        return self.from_data(self.data.detach())

    def roll(self, n: int):
        return self.from_data(self.data.roll(n, 0))

    @classmethod
    def from_variable(cls, variable):
        return cls(variable.G[0], variable.data)

    def from_data(self, data):
        return self.__class__(self.G[0], data)

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
        data = copy.deepcopy(self.data)
        return self.__class__(
            self.G[0],
            nn.Parameter(data) if isinstance(self.data, nn.Parameter) else data,
        )

    def edit(self, data):
        return self.__class__(self.G[0], self.data + data)

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

    def to_image(self, const_noise: bool = True, truncation: float = 1.0):
        return self.styles_to_image(
            self.to_styles(), const_noise, truncation=truncation
        )

    def styles_to_image(
        self, styles, const_noise: bool = True, truncation: float = 1.0
    ):
        mu = self.G[0].mapping.w_avg.reshape(1, 1, -1)
        truncation = torch.as_tensor(truncation, device=mu.device)

        if truncation.ndim != 0:
            assert truncation.ndim == 1
            assert len(truncation) == self.G[0].mapping.num_ws
            truncation = truncation.repeat_interleave(self.G[0].num_required_vectors())

        return (
            self.G[0].synthesis(
                mu.lerp(styles, truncation.reshape(1, -1, 1)),
                noise_mode="const" if const_noise else "random",
                force_fp32=True,
            )
            + 1.0
        ) / 2.0

    def before_step(self):
        pass

    def after_step(self):
        pass

    def init_to_target(self, target):
        pass

    def penalty(self, pred, target):
        return torch.tensor([0.0]).cuda()
