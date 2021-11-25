from .prelude import *
from .variables import *
from .criterions import *
from .jittering import *
from .io import *
from .inversion import *

import training.networks as networks

import matplotlib.pyplot as plt

def generator_with_len(self):
    pass


@dataclass(eq=False)
class Inverter:
    G: networks.Generator
    num_steps: int
    variable_type: Type[Variable]
    criterion: InversionCriterion = VGGCriterion()
    constraints: List[OptimizationConstraint] = None
    snapshot_frequency: int = 50
    seed: Optional[int] = 0
    create_schedule: None = None
    create_optimizer: None = None
    ema_weight: float = 0.95
    penalty: None = None
    # step_every_n: int = 0.1

    def __len__(self):
        return self.num_steps // self.snapshot_frequency

    def __post_init__(self):
        pass

    @torch.enable_grad()
    def all_inversion_steps(self, target) -> Iterable[Inversion]:
        if self.seed is not None:
            torch.manual_seed(self.seed)

        variables = []
        losses = []
        preds = []

        variable = self.sample_var(target)
        variables.append(variable)

        try:
            for i, loss in enumerate(self.inversion_loop(target, variable)):
                losses.append(loss)

                is_snapshot = i % self.snapshot_frequency == 0
                if is_snapshot:
                    variables.append(variable)
                    with torch.no_grad():
                        preds.append(variable.to_image())

                yield Inversion(target, variables, losses, preds, self.ema), is_snapshot
        except KeyboardInterrupt:
            pass
        
        return Inversion(target, variables, losses, preds, self.ema, final=True), True

    def __call__(self, *args, **kwargs):
        for inversion, _ in self.all_inversion_steps(*args, **kwargs):
            pass
        return inversion

    def sample_var(self, target):
        if isinstance(self.variable_type, Variable):
            return self.variable_type.copy()
        else:
            return self.variable_type.sample_from(self.G, len(target))

    def inversion_loop(self, target, variable) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        
        optimizer = self.create_optimizer(list(variable.parameters()))
        self.ema = variable.copy()

        if self.create_schedule is not None: # todo use a dummy schedule instead
            schedule = self.create_schedule(optimizer)
        else:
            schedule = None

        loss = None

        def compute_loss(do_backprop: bool = True):
            nonlocal loss
            pred = variable.to_image()
            loss = self.criterion(pred, target)

            expected_loss = loss.mean()

            if self.penalty is not None:
                expected_loss += self.penalty(pred)
            
            if loss.grad_fn is not None:
                expected_loss.backward()
            
            variable.before_step()

            return expected_loss

        for i in range(self.num_steps + 1):
            optimizer.zero_grad()
            optimizer.step(compute_loss)

            # if i % self.step_every_n == 0:
            if schedule is not None:
                schedule.step()
            variable.after_step()

            #!!!
            # with torch.no_grad():
            #     self.ema.data.copy_((1.0 - self.ema_weight) * variable.data.detach().clone() + self.ema_weight * self.ema.data)
            
            optimizer.zero_grad()

            yield loss.detach()


# todo pass in init as an object
# class CloseInitInverter(Inverter):
#     def sample_var(self, target):
#         return self.variable_type.find_init_point(self.G, target, VGGCriterion())


# class KDTreeInitInverter(Inverter):
#     def sample_var(self, target):
#         return self.variable_type.find_init_point(self.G, target, VGGCriterion())
