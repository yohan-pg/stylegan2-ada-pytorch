from ..prelude import *

from .eval import *


class EvalImageEditingConsistency(Evaluation):
    name: ClassVar[str] = "Image Editing Consistency"

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
            edited_var = inversion.final_variable + edit
            edited_image = edited_var.to_image()
            
            second_inversion = inverter(edited_image)
            unedited_var = second_inversion.final_variable - edit
            unedited_image = unedited_var.to_image()

            losses.append(inverter.criterion(unedited_image, inversion.target))

            save_image(
                torch.cat(
                    (
                        inversion.target,
                        unedited_image,
                    )
                ),
                f"{self.out_dir}/{experiment_name}/{i}.png",
                nrow=len(inversion.target),
            )
            save_image(
                torch.cat(
                    (
                        inversion.final_pred,
                        ((inversion.final_variable + edit) - edit).to_image(),
                    )
                ),
                f"{self.out_dir}/{experiment_name}/sanity_check{i}.png",
                nrow=len(inversion.target),
            )
            save_image(
                torch.cat(
                    (
                        inversion.target,
                        inversion.final_pred,
                        edited_image,
                        second_inversion.final_pred,
                        unedited_image,
                        (unedited_var * 1000.0).to_image(),
                        (unedited_var * 0.1).to_image()
                    )
                ),
                f"{self.out_dir}/{experiment_name}/trace_{i}.png",
                nrow=len(inversion.target),
            )
            save_image(
                target_2_var.to_image(),
                f"{self.out_dir}/{experiment_name}/edit_{i}.png",
                nrow=len(inversion.target),
            )

        return {"losses": torch.cat(losses)}

    def create_artifacts(
        self, dataloaders: Dict[str, InvertedDataloader]
    ):
        results = self.load_results_from_disk()
        self.make_table(results)
        self.make_histogram(results)
