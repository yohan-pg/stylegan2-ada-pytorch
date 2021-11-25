from .variable import *
from .w_variable import *


def slerp(a, b, t):
    "StyleGAN's slerp code, slightly modified so that norms are endpoints preserved and linearly interpolated between at intermediate steps."
    na = a.norm(dim=-1, keepdim=True)
    nb = b.norm(dim=-1, keepdim=True)
    a = a / na
    b = b / nb
    d = (a * b).sum(dim=-1, keepdim=True)
    p = t * torch.acos(d)
    c = b - d * a
    c = c / c.norm(dim=-1, keepdim=True)
    d = a * torch.cos(p) + c * torch.sin(p)
    d = d / d.norm(dim=-1, keepdim=True)
    return d * ((1.0 - t) * na + t * nb)


class ZVariable(Variable):
    space_name = "Z"
    default_lr = 0.03
    truncation = 1.0
    noise_amount = 0.0

    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1, init_scale=1.0):
        return cls(
            G,
            nn.Parameter(
                (
                    torch.randn(batch_size, G.num_required_vectors(), G.z_dim).cuda()
                ).squeeze(1)
                * init_scale
            ),
        )

    @classmethod
    def sample_random_from(cls, G: nn.Module, *args, **kwargs):
        return cls.sample_from(G, *args, **kwargs)

    def interpolate(self, other: "ZVariable", alpha: float) -> Variable:
        assert self.G == other.G

        var = self.__class__(
            self.G[0],
            slerp(
                self.data,
                other.data,
                alpha,
            ),
        )
        var.training = self.training
        return var

    # def interpolate(self, other: "ZVariable", alpha: float) -> Variable:
    #     assert self.G == other.G

    #     return self.__class__(
    #         self.G[0],
    #         slerp(
    #             self.data.reshape(self.data.shape[0], -1).reshape(self.data.shape),
    #             other.data.reshape(other.data.shape[0], -1).reshape(other.data.shape),
    #             alpha,
    #         ),
    #     )

    # interpolate = WVariable.interpolate

    def to_W(self) -> WVariable:
        return WVariable(
            self.G[0], self.to_styles()[:, : self.G[0].num_required_vectors()]
        )

    def interpolate_in_W(self, other: "ZVariable", alpha: float) -> Variable:
        return self.to_W().interpolate(other.to_W(), alpha)

    def to_styles(self, data=None):
        return self.G[0].mapping(
            self.data + torch.randn_like(self.data) * self.noise_amount if data is None else data, None
        )


def renormalize(x):
    norm = x.norm(dim=(1, 2), keepdim=True)
    target = math.sqrt(x.shape[1]) * math.sqrt(x.shape[2])
    return x / (norm + 1e-8) * target


def make_ZVariableWithDropoutOnW(p):
    class ZVariableWithDropoutOnW(ZVariable):
        dropout = nn.Dropout(p)

        def to_styles(self):
            styles = super().to_styles()

            G = self.G[0]
            mean = G.mapping.w_avg.reshape(1, 1, G.w_dim)

            if self.training:
                return self.dropout(styles - mean) + mean
            else:
                return styles

    return ZVariableWithDropoutOnW


# def make_ZVariableWithDropoutOnZ(p):
#     class ZVariableWithDropoutOnZ(ZVariable):
#         dropout = nn.Dropout(p)
#         truncation_factor = 1.0

#         def after_step(self):
#             norm = self.data.norm(dim=(1, 2), keepdim=True)
#             target = (
#                 math.sqrt(self.data.shape[1])
#                 * math.sqrt(self.data.shape[2])
#                 * self.truncation_factor
#             )

#             with torch.no_grad():
#                 self.data.copy_(self.data / (norm + 1e-10) * target)

#         def to_styles(self):
#             return self.G[0].mapping(self.dropout(self.data), None)

#     return ZVariableWithDropoutOnZ


def make_ZVariableWithRelativeDropoutOnZ(p):
    class ZVariableWithRelativeDropoutOnZ(ZVariable):
        dropout = nn.Dropout(p)
        truncation_factor = 1.0

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            with torch.no_grad():
                self.init_point = self.data.detach().clone()
                self.data.copy_(self.data * 0)

        # def after_step(self):
        #     norm = self.data.norm(dim=(1, 2), keepdim=True)
        #     target = (
        #         math.sqrt(self.data.shape[1])
        #         * math.sqrt(self.data.shape[2])
        #         * self.truncation_factor
        #     )

        #     with torch.no_grad():
        #         self.data.copy_(self.data / (norm + 1e-10) * target)

        def to_styles(self):
            return self.G[0].mapping(self.dropout(self.data) + self.init_point, None)

    return ZVariableWithRelativeDropoutOnZ


def make_ZVariableWithDropoutOnZ(p):
    class ZVariableWithDropoutOnZ(ZVariable):
        prob = p
        dropout = nn.Dropout(prob)
        truncation_factor = 1.0
        # decay = d

        def after_step(self):
            # self.prob *= self.decay
            # self.dropout = nn.Dropout(self.prob)
            pass
            # norm = self.data.norm(dim=(1, 2), keepdim=True)
            # target = (
            #     math.sqrt(self.data.shape[1])
            #     * math.sqrt(self.data.shape[2])
            #     * self.truncation_factor
            # )

            # with torch.no_grad():
            #     self.data.copy_(self.data / (norm + 1e-10) * target)

        def to_styles(self):
            return self.G[0].mapping(self.dropout(self.data), None)

    return ZVariableWithDropoutOnZ


def make_ZVariableWithAlphaDropoutOnZ(p):
    class ZVariableWithAlphaDropoutOnZ(ZVariable):
        prob = p
        dropout = nn.AlphaDropout(prob)
        truncation_factor = 1.0
        # decay = d

        def after_step(self):
            # self.prob *= self.decay
            # self.dropout = nn.Dropout(self.prob)
            pass
            # norm = self.data.norm(dim=(1, 2), keepdim=True)
            # target = (
            #     math.sqrt(self.data.shape[1])
            #     * math.sqrt(self.data.shape[2])
            #     * self.truncation_factor
            # )

            # with torch.no_grad():
            #     self.data.copy_(self.data / (norm + 1e-10) * target)

        def to_styles(self):
            return self.G[0].mapping(self.dropout(self.data), None)

    return ZVariableWithAlphaDropoutOnZ


def make_ZVariableWithFeatureAlphaDropoutOnZ(p, g=1.0):
    class ZVariableWithFeatureAlphaDropoutOnZ(ZVariable):
        prob = p
        dropout = nn.FeatureAlphaDropout(prob)
        truncation_factor = 1.0
        gamma = g
        # decay = d

        num_steps = 0

        def after_step(self):
            self.dropout.p *= self.gamma
            # self.prob *= self.decay
            # self.dropout = nn.Dropout(self.prob)
            pass
            # norm = self.data.norm(dim=(1, 2), keepdim=True)
            # target = (
            #     math.sqrt(self.data.shape[1])
            #     * math.sqrt(self.data.shape[2])
            #     * self.truncation_factor
            # )

            # with torch.no_grad():
            #     self.data.copy_(self.data / (norm + 1e-10) * target)

        def to_styles(self):
            return self.G[0].mapping(self.dropout(self.data), None)

    return ZVariableWithFeatureAlphaDropoutOnZ


def make_ZVariableWithDropoutOnZClamped(p):
    class ZVariableWithDropoutOnZ(ZVariable):
        prob = p
        dropout = nn.Dropout(prob)
        truncation_factor = 1.0
        # decay = d

        def after_step(self):
            # self.prob *= self.decay
            # self.dropout = nn.Dropout(self.prob)
            pass
            # norm = self.data.norm(dim=(1, 2), keepdim=True)
            # target = (
            #     math.sqrt(self.data.shape[1])
            #     * math.sqrt(self.data.shape[2])
            #     * self.truncation_factor
            # )

            # with torch.no_grad():
            #     self.data.copy_(self.data / (norm + 1e-10) * target)

        def to_styles(self):
            return self.G[0].mapping(self.dropout(torch.tanh(self.data) / 10000), None)

    return ZVariableWithDropoutOnZ


def make_ZVariableWithFakeDropoutOnZ(p):
    class ZVariableWithFakeDropoutOnZ(ZVariable):
        truncation_factor = 1.0

        @torch.no_grad()
        def before_step(self):
            self.backup = self.data.clone()

        @torch.no_grad()
        def after_step(self):
            mask = torch.rand_like(self.data) > p
            print(mask.sum())
            self.data.copy_((self.data - self.backup) * mask + self.backup)

    return ZVariableWithFakeDropoutOnZ


def make_ZVariableWithMultiplicativeNoise(p):
    class ZVariableWithMultiplicativeNoise(ZVariable):
        truncation_factor = 1.0

        def after_step(self):
            norm = self.data.norm(dim=(1, 2), keepdim=True)
            target = (
                math.sqrt(self.data.shape[1])
                * math.sqrt(self.data.shape[2])
                * self.truncation_factor
            )

            with torch.no_grad():
                self.data.copy_(self.data / (norm + 1e-10) * target)

        def to_styles(self):
            if self.training:
                data = self.data * (1 - torch.rand_like(self.data) * p)
            else:
                data = self.data
            return self.G[0].mapping(data, None)

    return ZVariableWithMultiplicativeNoise


class Z2Variable(ZVariable):
    space_name = "Z2"
    default_lr = 0.03

    split_point = 6
    gain = 1.0

    num_frozen_steps = 0
    current_step = 0

    randomize_later = False
    randomize_early = False

    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        z = normalize_2nd_moment(
            torch.randn(batch_size, G.num_required_vectors(), G.z_dim).cuda()
        ).squeeze(1)
        result = cls(
            G,
            nn.Parameter(torch.cat((z, z / cls.gain), dim=1)),
        )
        result.after_step()
        return result

    def after_step(self):
        self.current_step += 1

        z1, z2 = self.data.chunk(2, dim=1)

        with torch.no_grad():
            self.data.copy_(
                torch.cat(
                    (renormalize(z1), renormalize(z2 * self.gain) / self.gain), dim=1
                )
            )

    def to_styles(self):
        z1, z2 = self.data.chunk(2, dim=1)

        if self.current_step <= self.num_frozen_steps:
            z2 = z2.detach()

        L = self.split_point * self.G[0].num_required_vectors()
        return torch.cat(
            (
                self.G[0].mapping(torch.randn_like(z1) if self.randomize_early else z1, None)[:, :L],
                self.G[0].mapping(torch.randn_like(z2) if self.randomize_later else z2 * self.gain, None)[:, L:],
            ),
            dim=1,
        )


def make_ZVariableWithNoise(amount):
    class ZVariableWithNoise(ZVariable):
        space_name = "Zn"
        noise_gain = amount

        def to_styles(self):
            return self.G[0].mapping(
                self.data + torch.randn_like(self.data) * self.noise_gain, None
            )

    return ZVariableWithNoise


class WVariableConstrainToTypicalSetAllVecs(WVariable):
    truncation_factor = 1.0

    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        return cls(
            G,
            nn.Parameter(
                torch.randn(batch_size, G.num_required_vectors(), G.w_dim).cuda()
            ),
        )

    def after_step(self, diff):
        delta = self.data
        norm = delta.norm(dim=(1, 2), keepdim=True)
        target = (
            math.sqrt(self.data.shape[1])
            * math.sqrt(self.data.shape[2])
            * self.truncation_factor
        )

        with torch.no_grad():
            self.data.copy_(delta / (norm + 1e-8) * target)

    interpolate = ZVariable.interpolate

    def to_W(self):
        return WVariable(self.G[0], self.data)

    def to_styles(self) -> Styles:
        self.after_step(None)
        return super().to_styles() + self.G[0].mapping.w_avg


class WVariableConstrainToTypicalSet(WVariable):
    truncation_factor = 1.0

    sample_from = WVariable.sample_random_from

    interpolate = ZVariable.interpolate

    def after_step(self):
        norm = self.data.norm(dim=-1, keepdim=True)
        target = math.sqrt(self.data.shape[2]) * self.truncation_factor

        with torch.no_grad():
            self.data.copy_(self.data / (norm + 1e-8) * target)

    def to_W(self):
        return WVariable(self.G[0], self.data + self.G[0].mapping.w_avg)

    def to_styles(self) -> Styles:
        self.after_step(None)
        return super().to_styles() + self.G[0].mapping.w_avg


class ZAndWVariableConstrainToTypicalSet(ZVariable):
    truncation_factor = 0.72
    noise_gain = 0.0

    def project(self, x):
        if x.ndim == 3:
            norm = x.norm(dim=-1, keepdim=True)
            target = math.sqrt(x.shape[2]) * self.truncation_factor
        else:
            norm = x.norm(dim=(1), keepdim=True)
            target = math.sqrt(x.shape[1]) * self.truncation_factor
        return x / (norm + 1e-8) * target

    def after_step(self, diff):
        with torch.no_grad():
            self.data.copy_(
                self.project(self.data + torch.randn_like(self.data) * self.noise_gain)
            )

    def to_styles(self) -> Styles:
        delta = super().to_styles() - self.G[0].mapping.w_avg
        norm = delta.norm(dim=-1, keepdim=True)
        target = math.sqrt(delta.shape[2]) * self.truncation_factor
        return (delta / (norm + 1e-8) * target) + self.G[0].mapping.w_avg


class WVariableAroundMean(WVariableConstrainToTypicalSet):
    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        return cls(
            G,
            nn.Parameter(
                torch.randn(batch_size, G.num_required_vectors(), G.w_dim).cuda()
            ),
        )

    @classmethod
    def sample_random_from(cls, G: nn.Module, batch_size: int = 1, **kwargs):
        data = (
            torch.randn(batch_size, G.num_required_vectors(), G.z_dim).squeeze(1).cuda()
        )

        var = cls(
            G,
            nn.Parameter(data),
        )
        var.after_step(None)
        return var


def WithLearntTransform(cls):
    class VariableWithTransform(cls):
        gain = 0.1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.transform = nn.Parameter(torch.zeros(len(self.data), 3).cuda())

        def to_image(self):
            image = super().to_image()
            print(self.transform[-1])
            return (self.transform.unsqueeze(2).unsqueeze(3) * self.gain + 1) * image

    return VariableWithTransform
