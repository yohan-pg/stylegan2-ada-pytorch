from .prelude import *


class Variable(ToStyles, ABC, nn.Module):
    @final
    def __init__(self, G, params: torch.Tensor):
        super().__init__()
        self.G = [G]
        self.params = nn.Parameter(params)  # todo assert shape

    @abstractclassmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        raise NotImplementedError

    @abstractmethod
    def interpolate(self, other: "Variable", alpha: float) -> Styles:
        raise NotImplementedError

    def copy(self):
        return self.__class__(self.G[0], self.params.clone())


class WVariable(Variable):
    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        return WVariable(
            G,
            G.mapping(
                (torch.randn(1, G.num_required_vectors(), G.z_dim).squeeze(1).cuda()),
                None,
            )[:, : G.num_required_vectors(), :],
        )

    def interpolate(self, other: "WVariable", alpha: float) -> Styles:
        assert self.G == other.G
        return self.__class__(self.G[0], (1.0 - alpha) * self.params + alpha * other.params)

    def to_styles(self) -> Styles:
        return self.params.repeat(1, self.G[0].num_ws, 1)


class ZVariable(Variable):
    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        return ZVariable(
            G,
            (
                torch.rand(batch_size, G.num_required_vectors(), G.z_dim)
                .cuda()
            ).squeeze(1), 
        )

    # def interpolate(self, other: "ZVariable", alpha: float) -> Styles:
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
            
    #     return ZVariable(
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


class _Plus:
    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        return cls(G, super().to_styles(super().sample_from(G)))

    def to_styles(self) -> Styles:
        return self.params


class WPlusVariable(_Plus, WVariable):
    pass


# class ZPlusVariable(_Plus, ZVariable):
#     pass


class _InitializeAtMean:
    @classmethod
    def sample_from(
        cls, G: nn.Module, num_samples: int = 1000, batch_size: int = 10
    ):  #! can't parameterize this
        # w_avg = torch.zeros(1, G.num_required_vectors(), G.z_dim).cuda()

        # num_batches = num_samples // batch_size
        # assert num_batches >= 1

        # for _ in range(num_batches):
        #     z_samples = (
        #         torch.randn(batch_size, G.num_required_vectors(), G.z_dim)
        #         .squeeze(1)
        #         .cuda()
        #     )  # [N, V, C]
        #     w_samples = G.mapping(z_samples, None)  # [N, V*L, C]
        #     w_samples = w_samples[:, : G.num_required_vectors(), :]  # [N, V, C]
        #     w_avg += torch.mean(w_samples, dim=0, keepdim=True) / num_batches
        return cls(G, G.mapping.w_avg.reshape(1, 1, G.w_dim).repeat(1, G.num_required_vectors(), 1))

    def to_styles(self) -> Styles:
        return super().to_styles()

    # todo build up these stats for noise...
    # w_std = (np.sum((w_samples - w_avg) ** 2) / num_samples) ** 0.5
    # w_opt = torch.tensor(
    #     torch.tensor(w_avg).repeat(1, G.num_ws, 1) if w_plus else w_avg,
    #     dtype=torch.float32,
    #     requires_grad=True,
    # ).cuda()


class WVariableInitAtMean(_InitializeAtMean, WVariable):
    pass


class WPlusVariableInitAtMean(_Plus, _InitializeAtMean, WVariable):
    pass
