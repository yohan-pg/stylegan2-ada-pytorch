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
    learning_rate: Optional[float] = None
    beta_1: float = 0.9
    beta_2: float = 0.999
    constraints: List[OptimizationConstraint] = None
    snapshot_frequency: int = 50
    seed: Optional[int] = 0
    penalties: list = field(default_factory=list)
    constraints: list = field(default_factory=list)
    fine_tune_G: bool = False
    step_every_n: int = 1
            
    def all_inversion_steps(self, target):
        if self.seed is not None:
            torch.manual_seed(self.seed)

        variables = []
        losses = []
        preds = []

        try:
            for i, (loss, pred, variable) in enumerate(self.inversion_loop(target)):
                losses.append(loss)

                if (i % self.snapshot_frequency == 0) or (i == self.num_steps - 1):
                    variables.append(variable)
                    preds.append(pred)

                yield Inversion(target, variables, losses, preds)
        except KeyboardInterrupt:
            pass

    def __call__(self, *args, **kwargs):
        for inversion in self.all_inversion_steps(*args, **kwargs):
            pass
        return inversion

    def sample_var(self, target):
        return self.variable_type.sample_from(self.G, len(target))

    def inversion_loop(self, target) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        variable = self.sample_var(target)
        
        optimizer = torch.optim.Adam(
            list(variable.parameters())
            + (list(self.G.synthesis.parameters()) if self.fine_tune_G else []),
            self.learning_rate or self.variable_type.default_lr,  
            (self.beta_1, self.beta_2)
        )

        for step in range(self.num_steps + 1):
            t = step / self.num_steps

            for constraint in self.constraints:
                constraint.update(t)

            styles = variable.to_styles()
            pred = variable.styles_to_image(styles)
            loss = self.criterion(pred, target)

            cost = loss
            for penalty in self.penalties:
                cost += penalty(variable, styles, pred, target, loss)
            
            do_step = (step + 1) % self.step_every_n == 0
            (cost / self.step_every_n).mean().backward()
            if do_step:
                optimizer.step()
                optimizer.zero_grad()
            
            # variable.inform(optimizer.state_dict()['state'][0]['exp_avg_sq'])

            yield loss.detach(), pred.detach(), variable.copy()


class CloseInitInverter(Inverter):
    def sample_var(self, target):
        return self.variable_type.find_init_point(self.G, target, VGGCriterion())