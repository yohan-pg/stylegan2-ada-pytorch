from ..prelude import *

from .eval import *


class EvalImageEditingConsistency(Evaluation):
    name: ClassVar[str] = "Image Editing Consisency"

    alpha: ClassVar[float] = 0.5

    @torch.no_grad()
    def eval(self, experiment_name: str, dataloader: InvertedDataloader) -> Result:
        losses = []
        inverter = dataloader.inverter

        for i, inversion in enumerate(dataloader):
            target_2_var = inverter.variable_type.sample_random_from(
                inverter.G, batch_size=len(inversion.target)
            )

            edit = (target_2_var - inversion.final_variable) * self.alpha
            edited_image = (inversion.final_variable + edit).to_image()

            unedited_image = (inversion.rerun.final_variable - edit).to_image()
            losses.append(inverter.criterion(unedited_image, inversion.target))

            save_image(
                torch.cat(
                    (
                        target_2_var.to_image(),
                        inversion.target,
                        inversion.final_pred,
                        edited_image,
                        inversion.rerun.final_pred,
                        unedited_image,
                    )
                ),
                f"{self.out_dir}/{experiment_name}/{i}.png",
                nrow=len(inversion.target),
            )

        return {"losses": torch.cat(losses)}

    def create_artifacts(
        self, dataloaders: Dict[str, InvertedDataloader], results: Results
    ):
        results = self.load_results_from_disk()
        self.make_table(results)
        self.make_histogram(results)
