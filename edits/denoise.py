from prelude import *


#! error image preview is a bit ugly
#! assumes knowledge of noise distribution. Can we do better? Poisson model?
#! noise amounts are shared over batch elements, which is undesired!
#! pasting is weird. We should not be pasting extra noise, yet doing so helps the results? => but after the change to soft constraints, it does't again...
#! I don't even know why this works...
# todo save a plot of the inferred noise amount over # of iteratins

@dataclass(eq=False)
class Denoise(Edit):
    encoding_weight: ClassVar[float] = 10.0
    truncation_weight: ClassVar[float] = 3.0

    noise_amount: float = 0.1
    infered_noise_amount: nn.Parameter = field(
        default_factory=lambda: nn.Parameter(torch.tensor(0.0).cuda())
    )

    def f(self, pred):
        return pred + torch.randn_like(pred) * self.infered_noise_amount

    def f_ground_truth(self, target):
        return target + torch.randn_like(target) * self.noise_amount

    def paste(self, pred, target): 
        return target


if __name__ == "__main__":
    run_edit_on_examples(Denoise())
