from .prelude import *
from .variables import *
from .criterions import *
from .jittering import *
from .io import *
from .inversion import *


@dataclass(eq=False)
class Inverter:
    G_or_E: networks.Generator
    num_steps: int
    variable: Union[Type[Variable], Variable]
    criterion: InversionCriterion = VGGCriterion()
    snapshot_frequency: int = 50
    seed: Optional[int] = 0
    create_schedule: None = None
    create_optimizer: None = None
    penalty: Callable = None
    extra_params: List[nn.Parameter] = field(default_factory=list)
    parallel: bool = True
    randomize: bool = False
    gradient_scale: float = 1000.0

    def __post_init__(self):
        if self.parallel:
            self.G_or_E = Parallelize(self.G_or_E)

    def __len__(self):
        return self.num_steps

    def __call__(self, *args, **kwargs):
        for inversion, _ in self.all_inversion_steps(*args, **kwargs):
            pass
        return inversion

    @torch.enable_grad()
    def all_inversion_steps(self, target) -> Iterable[Inversion]:
        if self.seed is not None:
            torch.manual_seed(self.seed)

        variables = []
        losses = []
        penalties = []
        preds = []

        variable = self.new_variable(target)

        try:
            for i, (loss, penalty, pred) in enumerate(self.inversion_loop(target, variable)):
                losses.append(loss)
                penalties.append(penalty)

                is_snapshot = i % self.snapshot_frequency == 0
                if is_snapshot:
                    variables.append(variable)
                    preds.append(pred)

                yield Inversion(
                    target, variables, losses, penalties, preds, eval=is_snapshot
                ), is_snapshot
        except KeyboardInterrupt:
            pass

        return Inversion(target, variables, losses, penalties, preds, eval=True), True

    def new_variable(self, target):
        if isinstance(self.variable, Variable):
            var = self.variable.copy()
        else:
            var = (
                self.variable.sample_random_from
                if self.randomize
                else self.variable.sample_from
            )(self.G_or_E, len(target))
            var.init_to_target(target)

        return var

    def inversion_loop(
        self, target, variable
    ) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        optimizer = self.create_optimizer(
            list(variable.parameters()) + list(self.extra_params)
        )

        if self.create_schedule is not None:
            schedule = self.create_schedule(optimizer)
        else:
            schedule = None

        for i in range(self.num_steps + 1):
            variable.before_step()

            optimizer.zero_grad()

            yield self.backward_pass(target, variable)

            if i == self.num_steps:
                return

            if schedule is not None:
                schedule.step()
            optimizer.step()
            
            variable.after_step()
            
    def backward_pass(self, target, variable):
        pred = variable.to_image()
        loss = self.criterion(pred, target)
        expected_loss = loss.mean()
        penalty = variable.penalty(pred, target)

        if self.penalty is not None:
            penalty += self.penalty(variable, pred, target)

        if expected_loss.grad_fn is not None:
            (self.gradient_scale * (expected_loss + penalty)).backward()

        return loss.detach(), penalty.detach(), pred.detach()