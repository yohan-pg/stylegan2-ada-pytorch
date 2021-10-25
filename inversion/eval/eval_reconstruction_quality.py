from ..prelude import *

from .eval import *


class EvalReconstructionQuality(Evaluation):
    name: ClassVar[str] = "Reconstruction Quality"

    @torch.no_grad()
    def eval(self, experiment_name: str, dataloader: InvertedDataloader) -> Result:
        losses = []

        for i, inversion in enumerate(dataloader):
            losses.append(torch.stack(inversion.losses))
            losses.append(torch.stack(inversion.rerun.losses))
            save_image(
                torch.cat(
                    (
                        inversion.target,
                        inversion.final_pred,
                        inversion.rerun.final_pred,
                    )
                ),
                f"{self.out_dir}/{experiment_name}/{i}.png",
                nrow=len(inversion.final_pred),
            )

        return {
            "losses_per_step": torch.cat(losses),
            "losses": torch.cat(losses)[:, -1],
        }

    def create_artifacts(
        self, target_dataloader: RealDataloader, 
    ) -> None:
        results = self.load_results_from_disk()
        self.make_table(results)
        self.make_plot(target_dataloader, results)

    @torch.no_grad()
    def make_plot(
        self,
        target_dataloader: RealDataloader, 
        results: Results,
    ) -> None:
        description = f"over {target_dataloader.num_images} images"

        plt.figure()
        plt.title(f"[min, Q1, median, Q3, max] {description}")
        plt.suptitle("Reconstruction Error Bounds & Quantiles per Optimization Step")

        cmap = plt.get_cmap("tab10")

        names = []
        for i, (experiment_name, result) in enumerate(results.items()):
            all_losses = result["losses_per_step"].t()
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
