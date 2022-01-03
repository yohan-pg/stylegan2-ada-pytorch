from prelude import *


#! we want upsampling interpolation (or no upsampling) for computing the loss, but nearest for the visualization?


@dataclass(eq=False)
class Upsample(Edit):
    encoding_weight: ClassVar[float] = 10.0
    truncation_weight: ClassVar[float] = 3.0

    scale: int = 8

    def f(self, pred):
        return F.interpolate(
            F.avg_pool2d(pred, self.scale),
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
        )


if __name__ == "__main__":
    run_edit_on_examples(Upsample())
