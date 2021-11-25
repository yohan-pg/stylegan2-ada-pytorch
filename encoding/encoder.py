from inversion import *
from .stylegan_encoder_network import *
from training.loss import StyleGAN2EncoderLoss

import itertools


class Encoder(nn.Module):
    def __init__(
        self,
        G,
        D,
        variable_type: Type[Variable],
        gain: float = 1.0,
        lr: float = 0.001,
        beta_1=0.0,
        beta_2=0.0,
        const_noise: bool = True,
        fine_tune_discriminator: bool = False,
        discriminator_weight: float = 0.1,
        distance_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.fine_tune_discriminator = fine_tune_discriminator
        self.G = [G]
        self.D = [D]
        self.network = StyleGANEncoderNet.configure_for(
            G,
            w_plus="+" in variable_type.space_name,
            predict_residual_from_mean=not issubclass(variable_type, ZVariable),
            **kwargs,
        ).cuda()
        self.mean = G.mapping.w_avg.reshape(1, 1, G.w_dim)

        if fine_tune_discriminator:
            D.train()
            D.requires_grad_(True)

        self.optimizer = torch.optim.Adam(
            [
                {"params": list(self.parameters())},
                {"params": list(D.parameters())},
            ],
            lr=lr,
            betas=(beta_1, beta_2),
        )

        self.variable_type = variable_type
        self.const_noise = const_noise
        self.gain = gain
        self.sg2_loss = StyleGAN2EncoderLoss(
            torch.device("cuda"),
            G.mapping,
            G.synthesis,
            D,
            style_mixing_prob=0.0,
            pl_weight=0.0,
        )
        self.discriminator_weight = discriminator_weight
        self.distance_weight = distance_weight

    def backprop_discr(self, real: torch.Tensor, preds: torch.Tensor, var: Variable):
        #!!! sync unimplemented
        # def g():
            # return self.network.block0.conv.weight.grad
        self.sg2_loss.accumulate_gradients_encoder("Dboth", real, var.detach(), sync=False, gain=self.discriminator_weight)
        self.sg2_loss.accumulate_gradients_encoder("Gboth", real, var, sync=False, gain=self.discriminator_weight)

    def forward(self, x: torch.Tensor) -> Variable:
        ws = self.network(x) * self.gain + self.mean
        var = self.variable_type(self.G[0], ws)
        var.after_step()
        return var

    def fit(self, loader: List[torch.Tensor], criterion: InversionCriterion):
        try:
            while True:
                for targets in iter(loader):
                    targets = targets.cuda()

                    var = self(targets)
                    preds = var.to_image(self.const_noise)
                    loss = self.distance_weight * criterion(preds, targets)

                    yield preds, targets, loss

                    self.optimizer.zero_grad()
                    loss.mean(dim=0).backward(retain_graph=self.fine_tune_discriminator)
                    if self.fine_tune_discriminator:
                        self.backprop_discr(targets, preds, var)
                    self.optimizer.step()
        except KeyboardInterrupt:
            pass

    def make_prediction_grid(
        self, preds: torch.Tensor, targets: torch.Tensor, *others
    ) -> torch.Tensor:
        assert len(preds) == len(targets)
        return make_grid(
            torch.cat((preds, targets, *others, preds - targets)), nrow=len(targets)
        )

    def make_interpolation_grid(self, targets: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            grids = []

            for a, b in itertools.combinations_with_replacement(
                self(targets).split_into_individual_variables(), 2
            ):
                if a is not b:
                    grids.append(Interpolation.from_variables(a, b).grid())

            return make_grid(torch.cat(grids), nrow=len(grids[0]))
