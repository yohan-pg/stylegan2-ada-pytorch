from inversion import *
from encoding.stylegan_encoder_network import *
from training.loss import StyleGAN2EncoderLoss

import itertools
import warnings

warnings.filterwarnings("ignore", r"(.|\n)*imbalance between your GPUs.*")
warnings.filterwarnings("ignore", r"(.|\n)*Using or importing the ABCs.*")


@persistence.persistent_class
class Encoder(nn.Module):
    def __init__(
        self,
        G,
        D,
        variable_type: Type[Variable],
        gain: float = 1.0, 
        lr: float = 0.001,
        beta_1: float = 0.0,
        beta_2: float = 0.999,
        const_noise: bool = True,
        discriminator_weight: float = 0.1,
        distance_weight: float = 1.0,
        mean_regularization_weight: float = 0.0,
        discriminate_interpolations: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.fine_tune_discriminator = discriminator_weight > 0.0
        self.G = [G]
        self.D = [D]
        self.network = nn.parallel.DataParallel(
            StyleGANEncoderNet.configure_for(
                G,
                w_plus="+" in variable_type.space_name,
                **kwargs,
            ).cuda(),
        )
        self.mean = G.mapping.w_avg.reshape(1, 1, G.w_dim)
        self.mean_regularization_weight = mean_regularization_weight
        self.discriminate_interpolations = discriminate_interpolations

        if self.fine_tune_discriminator:
            D.train()
            D.requires_grad_(True)
            
        self.optimizer = torch.optim.Adam(
            list(self.parameters()) + list(D.parameters()),
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
        # if self.discriminate_interpolations:
        #     var = var.interpolate(var.roll(1), torch.rand(len(var), 1, 1).to(var.data.device) / 2)
        #! sync is unimplemented; this class does not work with DistributedDataParallel
        self.sg2_loss.accumulate_gradients_encoder(
            "Dboth", real, var.detach(), sync=False, gain=self.discriminator_weight
        )
        self.sg2_loss.accumulate_gradients_encoder(
            "Gboth", real, var, sync=False, gain=self.discriminator_weight
        )

    def _forward(self, x: torch.Tensor) -> Variable:
        deltas = self.network(x) * self.gain
        var = self.variable_type(self.G[0], deltas + self.mean)
        return deltas, var
    
    def forward(self, x: torch.Tensor) -> Variable:
        _, var = self._forward(x)
        return var

    def evaluate(self, targets, criterion):
        deltas, var = self._forward(targets)
        preds = var.to_image(self.const_noise)
        return var, preds, self.distance_weight * criterion(preds, targets) + self.mean_regularization_weight * deltas.pow(2.0).mean()

    def fit(self, loader: List[torch.Tensor], criterion: InversionCriterion):
        try:
            while True:
                for targets in iter(loader):
                    targets = targets.cuda()

                    var, preds, loss = self.evaluate(targets, criterion)
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
            torch.cat((targets, preds, *others, preds - targets)), nrow=len(targets)
        )

    def make_interpolation_grid(
        self, targets: torch.Tensor, max_size=6
    ) -> torch.Tensor:
        assert targets.shape[0] >= 2
        with torch.no_grad():
            grids = []

            for a, b in itertools.combinations_with_replacement(
                self(targets).split_into_individual_variables()[:max_size], 2
            ):
                if a is not b:
                    grids.append(Interpolation.from_variables(a, b).grid())

            return make_grid(torch.cat(grids), nrow=len(grids[0]))
