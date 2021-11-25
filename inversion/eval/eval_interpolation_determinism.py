from ..prelude import *

from .eval import *


class EvalInterpolationDeterminism(Evaluation):
    name: ClassVar[str] = "Interpolation Determinism"

    alpha: ClassVar[float] = 0.5

    @torch.no_grad()
    def eval(self, experiment_name: str, dataloader: InvertedDataloader) -> Result:
        losses = []

        for i, inversion in enumerate(dataloader):
            for j, other_inversion in enumerate(dataloader):
                if i < j:
                    first_inversion = inversion
                    second_inversion = dataloader.inverter(first_inversion.target)
                    var = other_inversion.final_variable

                    first_interpolation = (
                        first_inversion.final_variable
                        .interpolate(var, self.alpha)
                        .to_image()
                    )
                    second_interpolation = (
                        second_inversion.final_variable
                        .interpolate(var, self.alpha)
                        .to_image()
                    )
                    losses.append(
                        dataloader.inverter.criterion(
                            first_interpolation, second_interpolation
                        )
                    )

                    def interleave(a, b):
                        return torch.stack((a, b), dim=1).flatten(
                            start_dim=0, end_dim=1
                        )

                    save_image(
                        torch.cat(
                            [
                                interleave(inversion.target, inversion.target),
                                interleave(
                                    first_inversion.final_pred,
                                    second_inversion.final_pred,
                                ),
                                interleave(first_interpolation, second_interpolation),
                                interleave(*([var.to_image()] * 2)),
                                interleave(other_inversion.target, other_inversion.target),
                            ]
                        ),
                        f"{self.out_dir}/{experiment_name}/{i}_{j}.png",
                        nrow=len(inversion.target) * 2,
                    )

        return {"losses": torch.cat(losses)}

    def create_artifacts(
        self, dataloaders: Dict[str, InvertedDataloader]
    ):
        results = self.load_results_from_disk()
        self.make_table(results)
        self.make_histogram(results)
