from .prelude import *
from .variables import *
from .criterions import *
from .jittering import *
from .io import *
from .inversion import * 

import training.networks as networks

import matplotlib.pyplot as plt


@dataclass(eq=False)
class Inverter:
    G: networks.Generator
    num_steps: int
    variable_type: Type[Variable]
    criterion: InversionCriterion = VGGCriterion()
    learning_rate: float = 0.1
    beta_1: float = 0.9
    beta_2: float = 0.999
    constraints: List[OptimizationConstraint] = None
    snapshot_frequency: int = 50
    seed: Optional[int] = 0

    def __call__(self, target, out_path=None):
        if self.seed is not None:
            torch.manual_seed(self.seed)

        return invert(
            self.G,
            target=target,
            variable=self.variable_type.sample_from(self.G, len(target)),
            out_path=out_path,
            num_steps=self.num_steps,
            criterion=self.criterion,
            snapshot_frequency=self.snapshot_frequency,
            optimizer_constructor=lambda params: torch.optim.Adam(
                params, lr=self.learning_rate, betas=(self.beta_1, self.beta_2),
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
            losses.append(loss)

            if i % snapshot_frequency == 0: # todo or we are in the last iter
                variables.append(variable.copy())
                preds.append(pred)
                if out_path is not None: # todo clean up, move elsewhere
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

    for step in range(num_steps + 1):
        t = step / num_steps

        for constraint in constraints:
            constraint.update(t)

        styles = variable.to_styles()
        pred = variable.styles_to_image(styles)
        losses = criterion(pred, target)
        optimizer.zero_grad()
        losses.mean().backward()
        optimizer.step()

        yield losses.detach(), pred.detach()


def snapshot(pred, target, out_path):
    save_image(torch.cat((target, pred, (target - pred).abs())), out_path, nrow = (3 if len(pred) == 1 else len(pred)))
