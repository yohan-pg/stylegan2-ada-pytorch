from inversion import *

from measurement import *


class InterpolationDeterminismMeasurement(Measurement):
    name: ClassVar[str] = "Reconstruction Quality"
    out_dir: ClassVar[str] = "eval/interpolation_determinism"

    alpha: ClassVar[float] = 0.5

    def measure(
        self,
        dataloader: InvertedDataloader,
        alpha: float = 0.5,
    ):
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
                        .interpolate(var, alpha)
                        .to_image()
                    )
                    second_interpolation = (
                        second_inversion.final_variable.to_W()
                        .interpolate(var, alpha)
                        .to_image()
                    )
                    losses.append(
                        dataloader.inverter.criterion(
                            first_interpolation, second_interpolation
                        )
                    )

                    def interleave(a, b):
                        return torch.stack((a, b), dim=1).flatten(start_dim=0, end_dim=1)

                    save_image(
                        torch.cat(
                            [
                                interleave(inversion.target, inversion.target),
                                interleave(
                                    first_inversion.final_pred,
                                    second_inversion.final_pred,
                                ),
                                interleave(first_interpolation, second_interpolation),
                                torch.cat([var.to_image()] * (2 * len(inversion.target))),
                                torch.cat(
                                    [interpolation_inversion.target[j : j + 1]]
                                    * (2 * len(inversion.target))
                                ),
                            ]
                        ),
                        f"{self.out_dir}/{i}_{j}.png",
                        nrow=len(inversion.target) * 2,
                    )

        return torch.cat(losses)

    def create_artifacts(self, results: Results):
        pass

run_interpolation_determinism = InterpolationDeterminismMeasurement()

    # def make_table(out_dir: str, results: Dict[str, torch.Tensor]):
    #     with open(f"{OUT_DIR}/table.txt", "w") as file:
    #         print("## Interpolation Determinism", file=file)
    #         for experiment_name, losses in results.items():
    #             print(
    #                 f"{experiment_name}: {losses.mean().item():.04f} +- {losses.std().item():.04f}",
    #                 file=file,
    #             )

    # def make_histogram(losses):
    #     plt.figure()
    #     plt.hist(losses.cpu(), torch.ones(10).cpu())
    #     plt.savefig(f"{OUT_DIR}/hist.png")

    # def run_interpolation_determinism(dataloaders: Dict[str, InvertedDataloader]):
    #     results = {}

    #     for experiment_name, dataloader in dataloaders.items():
    #         method_dir = OUT_DIR + "/" + experiment_name
    #         os.makedirs(method_dir, exist_ok=True)

    #         results[experiment_name] = measure_interpolation_determinism(
    #             method_dir,
    #             dataloader,
    #         )
    #         with open(f"{method_dir}/losses.pkl", "wb") as file:
    #             pickle.dump(results[experiment_name], file=file)

    #     make_table(results)
