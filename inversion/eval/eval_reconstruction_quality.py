from ..prelude import *

from .eval import *


class EvalReconstructionQuality(Evaluation):
    @torch.no_grad()
    def produce_images(self, experiment_name: str, dataloader: InvertedDataloader) -> str:
        images_path = f"{self.out_dir}/{experiment_name}"
        fresh_dir(f"{images_path}/raw_images")
        for i, inversion in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            save_image(
                torch.cat(
                    (
                        inversion.target,
                        inversion.final_pred,
                    )
                ),
                f"{images_path}/{i}.png",
                nrow=len(inversion.final_pred),
            )
            for j, image in enumerate(inversion.final_pred):
                save_image(
                    image,
                    f"{images_path}/raw_images/{i}_{j}.png",
                    nrow=len(inversion.final_pred),
                )

        return images_path

    def compute_metrics(self, dataloader: InvertedDataloader, images_path: str) -> dict:
        losses = []

        for inversion in tqdm.tqdm(dataloader, total=len(dataloader)):
            losses.append(torch.stack(inversion.losses, dim=1))
        
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