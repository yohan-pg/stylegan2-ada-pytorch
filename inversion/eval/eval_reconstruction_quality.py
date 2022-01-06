from ..prelude import *

from .eval import *


class EvalReconstructionQuality(Evaluation):
    name: ClassVar[str] = "Reconstruction Quality"

    @torch.no_grad()
    def eval(self, experiment_name: str, dataloader: InvertedDataloader) -> Result:
        losses = []

        for i, inversion in tqdm.tqdm(enumerate(dataloader)):
            losses.append(torch.stack(inversion.losses, dim=1))
            save_image(
                torch.cat(
                    (
                        inversion.target,
                        inversion.final_pred,
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
        self,
        target_dataloader: RealDataloader,
    ) -> None:
        results = self.load_results_from_disk()
        self.make_table(results)
        self.make_plot(target_dataloader, results)

    