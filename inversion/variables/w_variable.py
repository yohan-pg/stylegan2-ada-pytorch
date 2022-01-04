from .variable import *


class WVariable(Variable):
    space_name = "W"
    default_lr = 0.1

    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1, avg_samples=10_000):
        data = G.mapping.w_avg.reshape(1, 1, G.w_dim).repeat(
            batch_size, G.num_required_vectors(), 1
        )

        return cls(
            G,
            nn.Parameter(data),
        )

    def to_W(self):
        return self

    @classmethod
    def sample_random_from(cls, G: nn.Module, batch_size: int = 1, **kwargs):
        data = G.mapping(
            (
                torch.randn(batch_size, G.num_required_vectors(), G.z_dim)
                .squeeze(1)
                .cuda()
            ),
            None,
            skip_w_avg_update=True,
            **kwargs,
        )[:, : G.num_required_vectors(), :]

        return cls(
            G,
            nn.Parameter(data),
        )

    def interpolate(self, other: "WVariable", alpha: float) -> Styles:
        assert self.G == other.G
        return self.__class__(self.G[0], self.data.lerp(other.data, alpha))
    
    def to_styles(self) -> Styles:
        data = self.data

        return data.repeat(1, self.G[0].num_ws, 1)


class WVariableRandomInit(WVariable):
    init_truncation = 0.25

    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        return cls(
            G,
            nn.Parameter(
                WVariable.sample_random_from(
                    G, batch_size, truncation_psi=cls.init_truncation
                ).data
            ),
        )


def l2(a, b):
    return ((a - b).pow(2.0).sum(dim=(1, 2)) + 1e-20).sqrt().mean()


def l2_avg(a, b):
    return ((a - b).pow(2.0).sum(dim=-1) + 1e-20).sqrt().mean()


def add_soft_encoder_constraint(
    cls,
    alpha=1.0,
    truncation=0.0,
    paste=lambda x: x,
    encoder_init=True,
    gamma=1.0,
    distance=nn.MSELoss(), 
):
    class EncoderConstrained(cls):
        space_name = cls.space_name + "e"

        @classmethod
        def sample_from(cls, E: nn.Module, *args, **kwargs):
            E.G[0].E = E
            return super().sample_from(E.G[0], *args, **kwargs)

        @classmethod
        def sample_random_from(cls, E: nn.Module, *args, **kwargs):
            E.G[0].E = E
            return super().sample_random_from(E.G[0], *args, **kwargs)

        @torch.no_grad()
        def init_to_target(self, target):
            init_point = self.G[0].E(target).data

            if cls.__name__ == "WPlusVariable":
                init_point = init_point.repeat(1, self.G[0].mapping.num_ws, 1)

            if encoder_init:
                self.data.copy_(init_point)

        def penalty(self, pred, target):
            nonlocal alpha
            nonlocal truncation
            alpha *= gamma
            truncation *= gamma
            anchor = self.G[0].E(paste(self.to_image())).data.detach()

            if cls.__name__ == "WPlusVariable":
                anchor = anchor.repeat(1, self.G[0].mapping.num_ws, 1)
            
            return (alpha * distance(self.data, anchor) + truncation * distance(
                self.data,
                self.G[0]
                .mapping.w_avg.reshape(1, 1, -1)
                .repeat(self.data.size(0), self.data.size(1), 1),
            )).unsqueeze(0)

    return EncoderConstrained



def add_though_encoder_constraint(
    cls,
    paste=lambda x: x,
    encoder_init=True,
    distance=nn.MSELoss(), 
):
    class EncoderConstrained(cls):
        space_name = cls.space_name + "e"

        @classmethod
        def sample_from(cls, E: nn.Module, *args, **kwargs):
            E.G[0].E = E
            return super().sample_from(E.G[0], *args, **kwargs)

        @classmethod
        def sample_random_from(cls, E: nn.Module, *args, **kwargs):
            E.G[0].E = E
            return super().sample_random_from(E.G[0], *args, **kwargs)

        @torch.no_grad()
        def init_to_target(self, target):
            init_point = self.G[0].E(target).data

            if cls.__name__ == "WPlusVariable":
                init_point = init_point.repeat(1, self.G[0].mapping.num_ws, 1)

            if encoder_init:
                self.data.copy_(init_point)

        def to_styles(self):
            return (self.G[0].E(self.styles_to_image(
               cls.to_styles(self), True
            ))).to_styles()

    return EncoderConstrained

def mahalanobis_distance(M, eps=1e-8):
    def distance(pred, target):
        diff = pred - target
        return l2(pred @ M.w_cov_inv_sqrt, target @ M.w_cov_inv_sqrt)
        # return (((diff @ M.w_cov_inv) * diff).sum(dim=-1) + eps).sqrt().sum(dim=1).mean()
    return distance

def shitty_mahalanobis_distance(M, eps=1e-8):
    def distance(pred, target):
        return l2(pred / M.w_std, target / M.w_std)
    return distance

def add_soft_encoder_constraint_mahalanobis(
    cls,
    alpha=1.0,
    truncation=0.0,
    paste=lambda x: x,
    *args,
    **kwargs,
):
    class EncoderConstrained(
        add_soft_encoder_constraint(
            cls, alpha=alpha, truncation=truncation, *args, **kwargs
        )
    ):
        def penalty(self, pred, target):
            update_statistics(self.G[0].mapping)
            distance = mahalanobis_distance(self.G[0].mapping)

            loss = 0.0

            if alpha > 0.0:
                anchor = self.G[0].E(paste(self.to_image())).data.detach()

                if cls.__name__ == "WPlusVariable":
                    anchor = anchor.repeat(1, self.G[0].mapping.num_ws, 1)
                loss = alpha * distance(self.data, anchor)

            return loss + truncation * distance(
                self.data,
                self.G[0]
                .mapping.w_avg.reshape(1, 1, -1)
                .repeat(self.data.size(0), self.data.size(1), 1),
            )

    return EncoderConstrained


def add_soft_encoder_constraint_image_space(
    cls, alpha=1.0, truncation=0.0, paste=lambda x: x, *args, **kwargs
):
    class EncoderConstrained(
        add_soft_encoder_constraint(
            cls, alpha=alpha, truncation=truncation, *args, **kwargs
        )
    ):
        def penalty(self, pred, target):
            G = self.G[0]
            E = G.E

            image = self.to_image()
            with torch.no_grad():
                latent_image = E(image)
                reimage = latent_image.to_image()
                anchor = latent_image.data

            if cls.__name__ == "WPlusVariable":
                anchor = anchor.repeat(1, self.G[0].mapping.num_ws, 1)

            return alpha * nn.MSELoss()(image, reimage,) + truncation * nn.MSELoss()(
                self.data,
                self.G[0]
                .mapping.w_avg.reshape(1, 1, -1)
                .repeat(self.data.size(0), self.data.size(1), 1),
            )

    return EncoderConstrained


def add_hard_encoder_constraint(
    cls,
    alpha=1.0,
    truncation=0.0,
    paste=lambda x: x,
    encoder_init=True,
    gamma=1.0,
    towards_init_point=None,
):
    class EncoderConstrained(cls):
        space_name = cls.space_name + "e"

        @classmethod
        def sample_from(cls, E: nn.Module, *args, **kwargs):
            E.G[0].E = E
            return super().sample_from(E.G[0], *args, **kwargs)

        @classmethod
        def sample_random_from(cls, E: nn.Module, *args, **kwargs):
            E.G[0].E = E
            return super().sample_random_from(E.G[0], *args, **kwargs)

        @torch.no_grad()
        def init_to_target(self, target):
            init_point = self.G[0].E(target).data
            self.init_point = init_point

            if cls.__name__ == "WPlusVariable":
                init_point = init_point.repeat(1, self.G[0].mapping.num_ws, 1)

            if encoder_init:
                self.data.copy_(init_point)

        def before_step(self):
            super().before_step()
            nonlocal alpha
            nonlocal truncation
            alpha *= gamma
            truncation *= gamma

            with torch.no_grad():
                anchor = (
                    self.init_point
                    if towards_init_point
                    else self.G[0].E(paste(self.to_image())).data.detach()
                )

                if cls.__name__ == "WPlusVariable":
                    anchor = anchor.repeat(1, self.G[0].mapping.num_ws, 1)

                self.data.copy_(
                    self.data.lerp(anchor, alpha).lerp(
                        self.G[0].mapping.w_avg.reshape(1, 1, -1), truncation
                    )
                )

    return EncoderConstrained


def update_statistics(M, num_samples=10_000, eps=1e-3):
    if not hasattr(M, "w_cov"):
        with torch.no_grad():
            key = (
                "use_adaconv"
                if hasattr(M, "use_adaconv")
                else "sample_for_adaconv"
            )

            temp_beta = M.w_avg_beta
            temp_adaconv = getattr(M, key)

            setattr(M, key, False)
            M.w_avg_beta = 0.0
            samples = M(torch.randn(num_samples, M.z_dim).cuda(), None)[:, 0, :]
            w_mean = samples.mean(dim=0, keepdim=True)
            samples -= w_mean
            M.w_mean = w_mean.unsqueeze(0)

            M.w_avg_beta = temp_beta
            setattr(M, key, temp_adaconv)
            M.w_cov = (samples.t() @ samples) / len(samples) + torch.eye(
                512
            ).cuda() * eps
            
            M.w_std = samples.std(dim=0)
            M.w_cov_inv = M.w_cov.inverse()
            
            U, S, V = M.w_cov.svd()
            M.w_cov_inv_sqrt = U @ torch.diag_embed(S ** -0.5) @ V.t()
