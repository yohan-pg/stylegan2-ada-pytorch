from .prelude import *


class Variable(ToStyles, ABC):
    @abstractmethod
    def interpolate(self, other: "Variable", alpha: float) -> Styles:
        raise NotImplementedError


class WVariable(Variable):
    def __init__(self, G: nn.Module, initialize_at_mean: bool = False):
        assert not initialize_at_mean #!!
        self.params = (
            torch.randn(1, G.num_required_vectors(), G.w_dim).cuda().requires_grad_(True)
        )
        self.w_dim = G.w_dim

    def lerp(self, other, alpha: float) -> Styles:
        return (1.0 - alpha) * self.params + alpha * other.w

    def interpolate(self, other: "WVariable", alpha: float) -> Styles:
        return self.variable_to_styles(self.lerp(other, alpha))

    def variable_to_styles(self) -> Styles:
        return self.params.repeat(1, self.w_dim, 1)


class WPlusVariable(Variable):
    def __init__(self, G: nn.Module, initialize_at_mean: bool = True):
        super().__init__(G, initialize_at_mean=initialize_at_mean)
        self.w = WVariable.variable_to_styles(self.w).detach().requires_grad_(True)

    def interpolate(self, other: "WVariable", alpha: float) -> Styles:
        return self.lerp(other, alpha)

    def variable_to_styles(self, w: torch.Tensor = None) -> Styles:
        return self.w


class ZVariable(Variable):
    def __init__(self, G: nn.Module, plus: bool = False):
        self.mapping = [G.mapping]
        self.z = (
            normalize_2nd_moment(
                torch.randn(
                    1, G.num_required_vectors(), G.z_dim
                )
            )
            .cuda()
            .requires_grad_(True)
        )

    def slerp(self, other, alpha: float) -> Styles:
        print("Warning: slerp not implemented.")
        return (1.0 - alpha) * self.params + alpha * other.w

    def variable_to_styles(self):
        return self.mapping[0](self.z.squeeze(1), None)


class ZPlusVariable(Variable):
    def __init__(self, G: nn.Module, initialize_at_mean: bool = True):
        super().__init__(G, initialize_at_mean=initialize_at_mean)
        self.w = WVariable.variable_to_styles(self.w)

    def interpolate(self, other: "WVariable", alpha: float) -> Styles:
        return self.lerp(other, alpha)

    def variable_to_styles(self, w: torch.Tensor = None) -> Styles:
        return self.w


class InitializeAtMean(nn.Module):
    def __init__(self, variable: Variable, num_samples: int = 100):#! wrong #
        self.variable = variable

        print(f"Computing W midpoint and stddev using {num_samples} samples...")
        # z_samples = np.random.RandomState(123).randn(
        #     w_avg_samples, G.num_required_vectors(), G.z_dim
        # ).squeeze(1)
        # w_samples = G.mapping(
        #     torch.from_numpy(z_samples).to(device).squeeze(1), None
        # )  # [N, L, C]
        # w_samples = (
        #     w_samples[:, : G.num_required_vectors(), :].cpu().numpy().astype(np.float32)
        # )  # [N, 1, C]
        # w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
        # w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5
        # w_opt = torch.tensor(
        #     torch.tensor(w_avg).repeat(1, G.num_ws, 1) if w_plus else w_avg,
        #     dtype=torch.float32,
        #     device=device,
        #     requires_grad=True,
        # )  # pylint: disable=not-callable
        #     # torch.Size([1, 1, 512])
        # w_out = torch.zeros(
        #     [num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device
        # ) # torch.Size([100, 1, 512])

    def variable_to_style(self):
        raise NotImplementedError
