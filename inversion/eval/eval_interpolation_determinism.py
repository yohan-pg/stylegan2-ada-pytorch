from ..prelude import *

from .eval import *


class EvalInterpolationDeterminism(Evaluation):
    name: ClassVar[str] = "Interpolation Determinism"

    alpha: ClassVar[float] = 0.5

    def eval(self, experiment_name: str, dataloader: InvertedDataloader) -> Result:
        losses = []

        for i, (inversion, interpolation_inversion) in tqdm.tqdm(
            enumerate(grouper(dataloader, 2))
        ):
            if interpolation_inversion is None:
                break

            first_inversion = inversion
            second_inversion = inversion.rerun

            with torch.no_grad():
                for j, var in enumerate(
                    interpolation_inversion.final_variable.to_W().split_into_individual_variables()
                ):
                    first_interpolation = (
                        first_inversion.final_variable.to_W()
                        .interpolate(var, self.alpha)
                        .to_image()
                    )
                    second_interpolation = (
                        second_inversion.final_variable.to_W()
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
                                    [interpolation_inversion.target[j : j + 1]]
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
        self.make_table(results)
        self.make_histogram(results)
