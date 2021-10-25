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
                first_inversion = inversion
                second_inversion = inversion.rerun
                var = other_inversion.final_variable.to_W() # todo what about the rerun?

                first_interpolation = (
                    inversion.final_variable.to_W()
                    .interpolate(var, self.alpha)
                    .to_image()
                )
                second_interpolation = (
                    inversion.rerun.final_variable.to_W()
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
                            torch.cat(
                                [var.to_image()] * (2 * len(inversion.target))
                            ),
                            torch.cat(
                                [other_inversion.target[j : j + 1]]
                                * (2 * len(inversion.target))
                            ),
                        ]
                    ),
                    f"{self.out_dir}/{experiment_name}/{i}_{j}.png",
                    nrow=len(inversion.target) * 2,
                )

        return {"losses": torch.cat(losses)}

    def create_artifacts(
        self, dataloaders: Dict[str, InvertedDataloader], results: Results
    ):
        results = self.load_results_from_disk()
        self.make_table(results)
        self.make_histogram(results)
