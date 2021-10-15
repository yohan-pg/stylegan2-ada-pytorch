from .prelude import *
from .variables import *
from .criterions import *
from .jittering import *
from .io import *
import training.networks as networks

import matplotlib.pyplot as plt


@dataclass(eq=False)
class Inverter:
    G: networks.Generator
    num_steps: int
    variable_type: Type[Variable]
    criterion: InversionCriterion = VGGCriterion()
    learning_rate: float = 0.1
    beta_1: float = 0.0
    beta_2: float = 0.0
    constraints: List[OptimizationConstraint] = None
    snapshot_frequency: int = 10

    def __call__(self, target, out_path=None):
        return invert(
            self.G,
            target=target,
            variable=self.variable_type.sample_from(self.G, len(target)),
            out_path=out_path,
            num_steps=self.num_steps,
            criterion=self.criterion,
            snapshot_frequency=self.snapshot_frequency,
            optimizer_constructor=lambda params: torch.optim.Adam(
                params, lr=self.learning_rate, betas=(self.beta_1, self.beta_2)
            ),
            constraints=[],
        )


def invert(
    G,
    target: torch.Tensor,
    variable: Variable,
    criterion: InversionCriterion,
    num_steps: int,
    optimizer_constructor: Callable[[List[torch.Tensor]], torch.optim.Optimizer],
    snapshot_frequency: Optional[int],
    out_path: Optional[str],
    constraints: List[OptimizationConstraint],
    fine_tune_G: bool = False,
) -> "Inversion":
    if out_path is not None:
        directory = os.path.dirname(out_path)
        os.makedirs(directory, exist_ok=True)

    variables = []
    losses = []
    preds = []

    try:
        for i, (loss, pred) in enumerate(
            inversion_loop(
                G,
                target,
                variable,
                criterion,
                optimizer_constructor,
                num_steps,
                constraints,
                fine_tune_G,
            )
        ):
            losses.append(loss.item())

            if i % snapshot_frequency == 0:
                variables.append(variable.copy())
                preds.append(pred)
                if out_path is not None:  # todo clean up, move elsewhere
                    snapshot(pred, target, out_path)
    except KeyboardInterrupt:
        pass

    return Inversion(target, variables, losses, preds)


def inversion_loop(
    G,
    target: ImageTensor,
    variable: Variable,
    criterion: InversionCriterion,
    optimizer_constructor: Callable[[List[torch.Tensor]], torch.optim.Optimizer],
    num_steps: int,
    constraints: List[OptimizationConstraint],
    fine_tune_G: bool,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    optimizer = optimizer_constructor(
        list(variable.parameters())
        + (list(G.synthesis.parameters()) if fine_tune_G else [])
    )

    for step in tqdm.tqdm(range(num_steps + 1)):
        t = step / num_steps

        for constraint in constraints:
            constraint.update(t)

        styles = variable.to_styles()
        pred = variable.styles_to_image(styles)
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        yield loss, pred


def snapshot(pred, target, out_path):
    save_image(torch.cat((target, pred, (target - pred).abs())), out_path)


class Inversion:
    def __init__(
        self,
        target: nn.Module,
        variables: List[Variable],
        losses: List[float],
        preds: List[torch.Tensor],
    ):
        self.variables = variables
        self.target = target
        self.losses = losses
        self.preds = preds

        for var in self.variables:
            var.eval()

        self.final_variable = self.variables[-1]
        self.final_pred = self.preds[-1]

    @staticmethod
    def save_losses_plot(inversions: List["Inversion"], out_path: str) -> None:
        plt.figure()
        plt.title("Reconstruction loss per step")

        for name, inversion in reversed(inversions.items()):
            plt.plot(inversion.losses, label=name)
        plt.legend()

        plt.ylim(0, 0.5)
        plt.xlabel("Optimization steps")
        plt.ylabel("Reconstruction loss")
        plt.savefig(out_path)

    def save_optim_trace(self, out_path: str) -> None:
        save_image(torch.cat(self.preds), out_path)

    def save_to_video(self, outdir: str):
        raise NotImplementedError
        video = imageio.get_writer(
            f"{outdir}/proj.mp4", mode="I", fps=10, codec="libx264", bitrate="16M"
        )
        print(f'Saving optimization progress video "{outdir}/proj.mp4"')
        for variable in self.variables:
            synth_image = G.synthesis(variable.to_styles(), noise_mode="const")
            synth_image = (synth_image + 1) * (255 / 2)
            synth_image = (
                synth_image.permute(0, 2, 3, 1)
                .clamp(0, 255)
                .to(torch.uint8)[0]
                .cpu()
                .numpy()
            )
            video.append_data(
                np.concatenate([target, torch.Tensor, synth_image], axis=1)
            )
        video.close()
