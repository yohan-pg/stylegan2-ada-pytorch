from prelude import *

from kornia.color import rgb_to_lab, lab_to_rgb

#! it looks like the clamping is what causes the hue issues?

#! alignment isn't always great

@dataclass(eq=False)
class Colorize(Edit):
    encoding_weight: ClassVar[float] = 2.0
    truncation_weight: ClassVar[float] = 100.0 #!!

    chroma_loss_weight: float = 1e-4
    chroma_objective: float = 20.0

    def f(self, x):
        x = rgb_to_lab(x.clamp(min=0.0, max=1.0)) #! clamping is not ideal
        x[:, 1:, :, :] *= 0
        return lab_to_rgb(x, clip=False)

    # def penalty(self, variable, pred, target):
    #     chroma = rgb_to_lab(pred.clamp(min=0.0, max=1.0))[:, 1:, :, :]
    #     return (
    #         self.chroma_loss_weight
    #         * (chroma.norm(dim=1) - self.chroma_objective)
    #         .pow(2.0)
    #         .mean()
    #     )


if __name__ == "__main__":
    run_edit_on_examples(Colorize())
