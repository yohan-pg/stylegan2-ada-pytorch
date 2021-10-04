from .prelude import *


class Variable(ToStyles, ABC, nn.Module):
    @final
    def __init__(self, G, params: torch.Tensor): # todo rename "params" to "data" to avoid confusion
        super().__init__()
        self.G = [G]
        self.params = params  # todo assert shape

    @abstractclassmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        raise NotImplementedError

    @abstractmethod
    def interpolate(self, other: "Variable", alpha: float) -> "Variable":
        raise NotImplementedError

    def copy(self):
        params = self.params.clone()
        return self.__class__(self.G[0], nn.Parameter(params) if isinstance(params, nn.Parameter) else params)

    def to_image(self, const_noise: bool = True):
        return self.styles_to_image(self.to_styles(), const_noise)
    
    def styles_to_image(self, styles, const_noise: bool = True):
        return (
            self.G[0].synthesis(
                styles, noise_mode="const" if const_noise else "random"
            )
            + 1
        ) / 2

    def split_into_individual_variables(self):
        return [self.__class__(self.G[0], p.unsqueeze(0)) for p in self.params]


class WVariable(Variable):
    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        return WVariable(
            G,
            nn.Parameter(
                G.mapping(
                    (
                        torch.randn(batch_size, G.num_required_vectors(), G.z_dim)
                        .squeeze(1)
                        .cuda()
                    ),
                    None,
                    skip_w_avg_update=True
                )[:, : G.num_required_vectors(), :]
            ),
        )

    def interpolate(self, other: "WVariable", alpha: float) -> Styles:
        assert self.G == other.G
        return self.__class__(
            self.G[0], (1.0 - alpha) * self.params + alpha * other.params
        )

    def to_styles(self) -> Styles:
        return self.params.repeat(1, self.G[0].num_ws, 1)


class WConvexCombinationVariable(Variable):
    class Params(nn.Module):
        def __init__(self, points, coefficients):
            super().__init__()
            self.points = points
            self.coefficients = coefficients

        def clone(self):
            return WConvexCombinationVariable.Params(self.points, self.coefficients)

    @classmethod
    def sample_from(
        cls,
        G: nn.Module,
        batch_size: int = 1,
        num_points: int = 100_000,
        inflation: float = 1.0,
    ):
        with torch.no_grad():
            temp = G.mapping.sample_for_adaconv
            G.mapping.sample_for_adaconv = False
            points = G.mapping(torch.randn(num_points, G.z_dim).cuda(), None, skip_w_avg_update=True).mean(
                dim=1
            )
            G.mapping.sample_for_adaconv = temp
            points = points + inflation * (points - G.mapping.w_avg.unsqueeze(0))

        var = WConvexCombinationVariable(
            G,
            WConvexCombinationVariable.Params(
                points,
                nn.Parameter(
                    torch.randn(batch_size, G.num_required_vectors(), num_points).cuda()
                    / num_points
                ),
            ),
        )

        return var

    def interpolate(self, other: "WConvexCombinationVariable", alpha: float) -> Variable:
        assert self.G == other.G
        return self.__class__(
            self.G[0],
            WConvexCombinationVariable.Params(
                (1.0 - alpha) * self.params.points + alpha * other.params.points,
                (1.0 - alpha) * self.params.coefficients
                + alpha * other.params.coefficients,
            ),
        )

    def to_styles(self) -> Styles:
        B, V, N = self.params.coefficients.shape
        mix = F.softmax(self.params.coefficients, dim=2)
        return (
            (mix.reshape(B * V, N) @ self.params.points)
            .reshape(B, V, -1)
            .repeat(1, self.G[0].num_ws, 1)
        ).squeeze(1)


class ZVariable(Variable):
    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        return ZVariable(
            G,
            nn.Parameter(
                (
                    torch.randn(batch_size, G.num_required_vectors(), G.z_dim).cuda()
                ).squeeze(1)
            ),
        )

    # def interpolate(self, other: "ZVariable", alpha: float) -> Variable:
    #     assert self.G == other.G
    #     # print("Warning: slerp not implemented! Using lerp.")

    #     # Spherical interpolation of a batch of vectors.
    #     def slerp(a, b, t):
    #         a = a / a.norm(dim=-1, keepdim=True)
    #         b = b / b.norm(dim=-1, keepdim=True)
    #         d = (a * b).sum(dim=-1, keepdim=True)
    #         p = t * torch.acos(d)
    #         c = b - d * a
    #         c = c / c.norm(dim=-1, keepdim=True)
    #         d = a * torch.cos(p) + c * torch.sin(p)
    #         d = d / d.norm(dim=-1, keepdim=True)
    #         return d

    #     return self.__class__(
    #         self.G[0],
    #         slerp(
    #             self.params,
    #             other.params,
    #             alpha,
    #         ),
    #     )

    interpolate = WVariable.interpolate

    def to_styles(self):
        # with torch.no_grad(): #?
        #     self.params.copy_(normalize_2nd_moment(self.params, dim=2))
        return self.G[0].mapping(self.params, None) 


class YVariable(Variable):
    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        return YVariable(
            G,
            nn.Parameter(
                (
                    torch.zeros(batch_size, G.num_required_vectors(), G.z_dim).cuda()
                ).squeeze(1)
            ),
        )

    interpolate = ZVariable.interpolate

    def to_styles(self):
        return self.params.repeat(1, self.G[0].num_ws, 1)


class PlusVariable(ABC):
    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        return cls(G, nn.Parameter(super().to_styles(super().sample_from(G))))

    def to_styles(self) -> Styles:
        return self.params


class WPlusVariable(PlusVariable, WVariable):
    pass


class ZPlusVariable(PlusVariable, ZVariable):
    pass



class ConstrainToMean(ToStyles):
    def __init__(self, variable, tau):
        super().__init__()
        self.variable = variable
        self.tau = tau
    
    def to_image(self, const_noise: bool = True):
        return self.variable.styles_to_image(self.variable.to_styles(), const_noise)

    def styles_to_image(self, styles, const_noise: bool = True):
        return self.variable.styles_to_image(styles, const_noise)

    def copy(self):
        return ConstrainToMean(self.variable.copy(), self.tau)

    def _project(self, G, styles):
        avg = G.mapping.w_avg.reshape(1, 1, G.w_dim)
        diff = styles - avg
        norm = diff.norm(dim=2, keepdim=True)
        
        return avg + diff * self.tau

    def to_styles(self) -> Styles:
        return self._project(self.variable.G[0], self.variable.to_styles())
    
    def interpolate(self, other, alpha: float) -> Variable:
        var = self.variable.interpolate(other.variable, alpha)
        return ConstrainToMean(var, self.tau)
    
    # todo build up these stats for noise...
    # w_std = (np.sum((w_samples - w_avg) ** 2) / num_samples) ** 0.5
    # w_opt = torch.tensor(
    #     torch.tensor(w_avg).repeat(1, G.num_ws, 1) if w_plus else w_avg,
    #     dtype=torch.float32,
    #     requires_grad=True,
    # ).cuda()


# todo build up these stats for noise...
# w_std = (np.sum((w_samples - w_avg) ** 2) / num_samples) ** 0.5
# w_opt = torch.tensor(
#     torch.tensor(w_avg).repeat(1, G.num_ws, 1) if w_plus else w_avg,
#     dtype=torch.float32,
#     requires_grad=True,
# ).cuda()


class WVariableInitAtMean(WVariable):
    @classmethod
    def sample_from(
        cls, G: nn.Module, batch_size: int = 1
    ):  
        return cls(
            G,
            nn.Parameter(G.mapping.w_avg.reshape(1, 1, G.w_dim).repeat(
                1, G.num_required_vectors(), 1
            )),
        )


class ZVariableInitAtMean(WVariable):
    @classmethod
    def sample_from(
        cls, G: nn.Module, batch_size: int = 1
    ):  
        return cls(
            G,
            nn.Parameter(ZVariable.sample_from(G, batch_size).params * 0),
        )


class WVariableInitAtMeanTruncated(WVariableInitAtMean):

    # interpolate = ZVariable.interpolate
    
    def to_styles(self) -> Styles:
        styles = super().to_styles()
        # todo review that shapes work out (no broadcasting bug)
        # return normalize_2nd_moment(styles)
        w_avg = self.G[0].mapping.w_avg
        # styles = w_avg.lerp(styles, 0.1) #!!
        styles = styles / ((styles - w_avg).norm(dim=2, keepdim=True) + 1) * 10
        print((styles - w_avg).norm(dim=2).mean())
        return styles

class WPlusVariableInitAtMean(PlusVariable, WVariableInitAtMean):
    pass

