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
    # @final
    def __init__(self, G, data: torch.Tensor):
        super().__init__()
        self.G = [G]
        # if data.ndim == 3:
        #     pass
        #     # assert data.shape[1:] == (G.num_required_vectors(), G.w_dim)
        # else:
        #     assert data.ndim == 2
        #     # assert data.shape[1] == G.w_dim
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

    def to_image(self, const_noise: bool = True, truncation=1.0):
        return self.styles_to_image(self.to_styles(), const_noise, truncation=truncation)

    def disable_noise(self):
        return self

    def styles_to_image(self, styles, const_noise: bool = True, truncation=1.0):
        mu = self.G[0].mapping.w_avg.reshape(1, 1, -1)
        truncation = torch.as_tensor(truncation, device=mu.device)

        if truncation.ndim != 0:
            assert truncation.ndim == 1
            assert len(truncation) == self.G[0].mapping.num_ws
            truncation = truncation.repeat_interleave(self.G[0].num_required_vectors())
        
        return (
            self.G[0].synthesis(
                mu.lerp(styles, truncation.reshape(1, -1, 1)), 
                noise_mode="const" if const_noise else "random", force_fp32=True
            )
            + 1
        ) / 2

    def before_step(self):
        pass

    def after_step(self):
        pass

    def with_noise(self, amount: float):
        var = self.copy()

        with torch.no_grad():
            var.data += torch.randn_like(var.data) * amount

        return var

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
    def find_init_point(cls, G, target, criterion, num_iters=100, batch_size=16):
        best_var = cls.sample_random_from(G, len(target))
        best_distance = torch.ones(len(target), device=target.device) * float("Inf")

        for _ in tqdm.tqdm(range(num_iters)):
            var = cls.sample_random_from(G, batch_size)
            for sample, var in zip(
                var.to_image(), var.split_into_individual_variables()
            ):
                distance = criterion(sample.unsqueeze(0), target)
                mask = distance < best_distance
                best_distance = torch.where(mask, distance, best_distance)
                best_var.data.copy_(
                    torch.where(
                        mask.unsqueeze(1)
                        .unsqueeze(2)
                        .repeat(1, *best_var.data.shape[1:]),
                        var.data.repeat_interleave(len(target), dim=0),
                        best_var.data,
                    )
                )

        save_image(
            torch.cat((target, best_var.to_image())), "tmp.png", nrow=len(target)
        )

        return best_var


def TransformedVariable(cls, f):
    class TransformedVariable(cls):
        def to_image(self):
            return f(super().to_image())

    return TransformedVariable


def ClippedVariable(cls, clipping: float):
    class ClippedVariable(cls):
        def before_step(self):
            self.data.grad += torch.randn_like(self.data.grad) * 0.01

    return ClippedVariable


def color_match(pred, target, eps=1e-30):
    pred = pred - pred.mean(dim=(2, 3), keepdim=True)
    pred = pred / (pred.std(dim=(2, 3), keepdim=True) + eps)
    mu = target.mean(dim=(2, 3), keepdim=True)
    pred *= (target - mu).std(dim=(2, 3), keepdim=True)
    pred += mu
    return pred


class VariableSandwich:
    def __init__(self, Z, W):
        self.Z = Z
        self.W = W

    def to_image(self):
        self
