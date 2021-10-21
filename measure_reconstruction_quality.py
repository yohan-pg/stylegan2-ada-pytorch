from inversion import *

from measurement import *


class ReconstructionQualityMeasurement(Measurement):
    name: ClassVar[str] = "Reconstruction Quality"
    out_dir: ClassVar[str] = "eval/reconstruction_quality"

    def measure(self, dataloader: InvertedDataloader) -> LossesPerStep:
        all_losses = []

        for _, inversion in enumerate(dataloader):
            all_losses.append(torch.stack(inversion.losses, dim=1))

        return torch.cat(all_losses)

    def create_artifacts(self, results: Results):
        self.make_table(results)
        self.make_plot(results)

    @torch.no_grad()
    def make_plot(
        self,
        results: Dict[str, torch.Tensor],
        description: str = "",
    ):
        plt.figure()
        plt.title(f"[min, Q1, median, Q3, max] {description}")
        plt.suptitle("Reconstruction Error Bounds & Quantiles per Optimization Step")

        cmap = plt.get_cmap("tab10")

        names = []
        for i, (experiment_name, all_losses) in enumerate(results.items()):
            names.append(experiment_name)
            low = all_losses.amin(dim=0)
            q1 = all_losses.quantile(0.25, dim=0)
            medians = all_losses.median(dim=0).values
            q3 = all_losses.quantile(0.75, dim=0)
            high = all_losses.amax(dim=0)
            assert all(
                data.shape == all_losses[0].shape
                for data in [low, q1, medians, q3, high]
            )

            plt.plot(medians.cpu(), color=cmap(i))
            plt.fill_between(
                range(all_losses.shape[1]), q1.cpu(), q3.cpu(), alpha=0.3, color=cmap(i)
            )
            plt.fill_between(
                range(all_losses.shape[1]),
                low.cpu(),
                high.cpu(),
                alpha=0.2,
                color=cmap(i),
            )
            plt.ylim(0.0, 0.6)

        plt.legend(names)

        plt.xlabel("Optimization Step")
        plt.ylabel("Reconstruction Error")
        plt.ylim(0)
        plt.savefig(f"{self.out_dir}/plot.png")


run_reconstruction_quality = ReconstructionQualityMeasurement()

# def make_table(results: Dict[str, torch.Tensor]):
#     with open(f"{OUT_DIR}/table.txt", "w") as file:
#         print("## Reconstruction Quality", file=file)
#         for experiment_name, losses in results.items():
#             all_final_losses = losses[:, -1]
#             print(
#                 f"{experiment_name}: {all_final_losses.mean().item():.04f} +- {all_final_losses.std().item():.04f}",
#                 file=file,
#             )


# def run_reconstruction_quality(dataloaders: Dict[str, InvertedDataloader]):
#     results = {}

#     for experiment_name, dataloader in dataloaders.items():
#         os.makedirs(f"{OUT_DIR}/{experiment_name}", exist_ok=True)

#         results[experiment_name] = measure_reconstruction_quality(dataloader)

#         with open(f"{OUT_DIR}/{experiment_name}/losses.pkl", "wb") as file:
#             pickle.dump(results[experiment_name], file)

#     make_table(results)
#     make_plot(
#         results,
#         f"over {dataloader.num_images} images ({dataloader.inverter.variable_type.space_name})",
#     )
