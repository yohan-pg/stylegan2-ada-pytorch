from ..prelude import *

from .eval import *

from .fid import compute_fid


class EvalInterpolationRealism(Evaluation):
    table_stat: str = "FID"

    @torch.no_grad()
    def produce_images(
        self, experiment_name: str, dataloader: InvertedDataloader
    ):
        print("Generating interpolation images...")
        fakes_path = f"{self.out_dir}/{experiment_name}/fakes"

        for (i, j), (inversion, other_inversion) in tqdm.tqdm(
            self.all_image_pairs(dataloader), total=self.num_image_pairs(dataloader)
        ):
            midpoint = (
                inversion.final_variable
                .interpolate(other_inversion.final_variable, 0.5)
                .to_image()
            )

            os.makedirs(fakes_path, exist_ok=True)
            for k, image in enumerate(midpoint):
                save_image(
                    image.unsqueeze(0),
                    f"{fakes_path}/{i}_{j}_{k}.png",
                )

            save_image(
                torch.cat(
                    [
                        inversion.target,
                        inversion.final_pred,
                        inversion.final_variable
                        .interpolate(other_inversion.final_variable, 0.25)
                        .to_image(),
                        midpoint,
                        inversion.final_variable
                        .interpolate(other_inversion.final_variable, 0.75)
                        .to_image(),
                        other_inversion.final_pred,
                        other_inversion.target,
                    ]
                ),
                f"{self.out_dir}/{experiment_name}/{i}_{j}.png",
                nrow=len(inversion.final_pred),
            )

        return fakes_path

    def compute_metrics(self, dataloader: InvertedDataloader, images_path: str) -> dict:
        return {"FID": compute_fid(dataloader, images_path)}

    def all_image_pairs(self, dataloader: InvertedDataloader):
        for i, inversion in enumerate(dataloader):
            for j, other_inversion in enumerate(dataloader):
                if i < j:
                    yield (i, j), (inversion, other_inversion)

    def num_image_pairs(self, dataloader: InvertedDataloader):
        n = len(dataloader)
        return n * (n - 1) // 2

    def create_artifacts(self, dataloaders: Dict[str, InvertedDataloader]):
        results = self.load_results_from_disk()
        self.make_table(results)

