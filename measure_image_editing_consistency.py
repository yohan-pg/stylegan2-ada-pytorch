from inversion import *

from measurement import *

class ImageEditingConsistencyMeasurement(Measurement):
    name: ClassVar[str] = "Image Editing Consisency"
    our_dir: str = "eval/image_editing_consistency"

    def measure_image_editing_consistency(
        outdir: str,
        dataloader: Iterable[torch.Tensor],
        alpha: float = 0.5,
    ):
        losses = []
        inverter = dataloader.inverter

        for i, inversion in enumerate(dataloader):
            target_2_var = inverter.variable_type.sample_random_from(
                inverter.G, batch_size=len(inversion.target)
            )

            first_inversion = inversion

            with torch.no_grad():
                edit = (target_2_var - first_inversion.final_variable) * alpha
                edited_image = (first_inversion.final_variable + edit).to_image()

            second_inversion = inversion.rerun

            with torch.no_grad():
                unedited_image = (second_inversion.final_variable - edit).to_image()
                losses.append(inverter.criterion(unedited_image, inversion.target))

            save_image(
                torch.cat(
                    (
                        target_2_var.to_image(),
                        inversion.target,
                        first_inversion.final_pred,
                        edited_image,
                        second_inversion.final_pred,
                        unedited_image,
                    )
                ),
                f"{outdir}/{i}.png",
                nrow=len(inversion.target),
            )

        return torch.cat(losses)



    #         with open(f"{OUT_DIR}/{experiment_name}/losses.pkl", "wb") as file:
    #             pickle.dump(results[experiment_name], file=file)
    
    # def make_table(results):
    #     with open(f"{OUT_DIR}/table.txt", "w") as file:
    #         print("## Image Editing Consistency", file=file)
    #         for experiment_name, losses in results.items():
    #             print(f"{experiment_name}: {round(losses.mean().item(), 6)} +- {losses.std().item():.04f}", file=file)


    # def run_image_editing_consistency(dataloaders: Dict[str, InvertedDataloader]):
    #     results = {}

    #     for experiment_name, dataloader in dataloaders.items():
    #         experiment_dir = OUT_DIR + "/" + experiment_name
    #         os.makedirs(experiment_dir, exist_ok=True)

    #         results[experiment_name] = measure_image_editing_consistency(
    #             experiment_dir, dataloader
    #         )


    #     make_table(results)


run_image_editing_consistency = ImageEditingConsistencyMeasurement()