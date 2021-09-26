from .prelude import *
from .variables import *
from .criterions import *
from .jittering import *
from .io import *

import matplotlib.pyplot as plt


def invert(
    G,
    target: torch.Tensor,
    variable: Type[Variable],
    criterion: InversionCriterion,
    num_steps: int,
    optimizer_constructor: Callable[[List[torch.Tensor]], torch.optim.Optimizer],
    snapshot_frequency: Optional[int],
    out_path: Optional[str],
    constraints: List[OptimizationConstraint]
) -> "Inversion":
    # assert os.path.isfile(out_path) # todo
    directory = os.path.dirname(out_path)
    # if os.path.exists(directory):
    #     os.path.
    os.makedirs(directory, exist_ok=True)

    variables = []
    losses = []
    preds = []

    try:
        for i, (loss, pred) in enumerate(
            inversion_loop(G, target, variable, criterion, optimizer_constructor, num_steps, constraints)
        ):
            if i % snapshot_frequency == 0:
                variables.append(variable.copy())
                losses.append(loss.item())
                preds.append(pred)

                if out_path is not None:  # todo clean up, move elsewhere
                    snapshot(pred, target, out_path)
    except KeyboardInterrupt:
        pass

    return Inversion(target, variables, losses, preds)


def inversion_loop(
    G,
    target: ImageTensor,
    # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    variable: Variable,
    criterion: InversionCriterion,
    optimizer_constructor: Callable[[List[torch.Tensor]], torch.optim.Optimizer],
    num_steps: int,
    constraints: List[OptimizationConstraint]
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    
        optimizer = optimizer_constructor(variable.parameters())

        for step in tqdm.tqdm(range(num_steps + 1)):
            t = step / num_steps 

            for constraint in constraints:
                constraint.update(t)

            styles = variable.to_styles()
            styles = styles #+ 1 * torch.randn_like(styles) #!!
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

        self.final_variable = self.variables[-1]

    def save_losses_plot(self, out_path: str) -> None:
        plt.figure()
        plt.plot(self.losses)
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
