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
    variable_type: Type[Variable]
    criterion: InversionCriterion = VGGCriterion()
    snapshot_frequency: int = 50
    seed: Optional[int] = 0
    create_schedule: None = None
    create_optimizer: None = None
    penalty: Callable = None
    extra_params: List[nn.Parameter] = field(default_factory=list)
    parallel: bool = True
    randomize: bool = False
    gradient_scale: float = 1.0

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

                yield Inversion(
                    target, variables, losses, preds, eval=is_snapshot
                ), is_snapshot
        except KeyboardInterrupt:
            pass

        return Inversion(target, variables, losses, preds, eval=True), True

    def sample_var(self, target):
        if isinstance(self.variable_type, Variable):
            var = self.variable_type.copy()
        else:
            var = (self.variable_type.sample_random_from if self.randomize else self.variable_type.sample_from)(self.G_or_E, len(target))
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

        loss = None

        def compute_loss():
            nonlocal loss
            pred = variable.to_image()
            loss = self.criterion(pred, target)

            expected_loss = loss.mean()
            expected_loss += variable.penalty(pred, target)

            if self.penalty is not None:
                expected_loss += self.penalty(variable, pred, target)

            if loss.grad_fn is not None:
                (self.gradient_scale * expected_loss).backward()

            variable.before_step()

            return expected_loss

        for _ in range(self.num_steps + 1):
            optimizer.zero_grad()
            optimizer.step(compute_loss)

            variable.after_step()

            if schedule is not None:
                schedule.step()

            yield loss.detach()

