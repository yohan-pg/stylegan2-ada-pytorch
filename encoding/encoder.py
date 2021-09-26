from inversion import *
from .stylegan_encoder_network import *

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
        fine_tune_generator: bool = False,
        fine_tune_discriminator: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.fine_tune_generator = fine_tune_generator
        self.fine_tune_discriminator = fine_tune_discriminator
        self.G = [G]
        self.D = [D]
        self.network = StyleGANEncoderNet.configure_for(
            G, w_plus=issubclass(variable_type, PlusVariable), **kwargs
        ).cuda()
        self.mean = G.mapping.w_avg.reshape(1, 1, G.w_dim)
        if fine_tune_generator:
            G.requires_grad_(True)
        if fine_tune_discriminator:
            D.requires_grad_(True)
        self.optimizer = torch.optim.Adam(
            [
                {"params": list(self.parameters())},
                {
                    "params": list(G.parameters()) if fine_tune_generator else [],
                    "lr": lr, 
                },
            ],
            lr=lr,
            betas=(beta_1, beta_2),
        )

        self.variable_type = variable_type
        self.const_noise = const_noise
        self.gain = gain

        assert not self.fine_tune_discriminator, "Unimplemented"

    def discriminator_loss(self, real: torch.Tensor, fake: torch.Tensor):
        return

    def forward(self, x: torch.Tensor) -> Variable:
        ws = self.network(x) * self.gain + self.mean
        return self.variable_type(self.G[0], ws)

    def fit(self, loader: List[torch.Tensor], criterion: InversionCriterion):
        try:
            while True:
                for targets in iter(loader):
                    targets = targets.cuda()

                    preds = self(targets).to_image(self.const_noise)
                    loss = criterion(preds, targets)

                    yield preds, targets, loss

                    self.optimizer.zero_grad()
                    loss.backward()
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
                    grids.append(Interpolator(self.G[0]).interpolate(a, b).grid())

            return make_grid(torch.cat(grids), nrow=len(grids[0]))
