from ..prelude import *

from .eval import *

from .fid import compute_fid


class EvalInterpolationRealism(Evaluation):
    name: ClassVar[str] = "Interpolation Realism"

    table_stat: str = "FID"

    @torch.no_grad()
    def eval(self, experiment_name: str, dataloader: InvertedDataloader) -> Result:
        fakes_path = self.generate_interpolation_images(experiment_name, dataloader)
        return {"FID": compute_fid(dataloader, fakes_path)}

    def generate_interpolation_images(
        self, experiment_name: str, dataloader: InvertedDataloader
    ):
        print("Generating interpolation images...")
        fakes_path = f"{self.out_dir}/{experiment_name}/fakes"

        for (i, j, k), (inversion, other_inversion) in tqdm.tqdm(
            self.all_image_pairs(dataloader), total=self.num_image_pairs(dataloader)
        ):
            midpoint = (
                inversion.final_variable.roll(k)
                .interpolate(other_inversion.final_variable, 0.5)
                .to_image()
            )
            save_image(
                torch.cat(
                    [
                        inversion.target.roll(k, dims=[0]),
                        inversion.final_pred.roll(k, dims=[0]),
                        midpoint,
                        other_inversion.final_pred,
                        other_inversion.target,
                    ]
                ),
                f"{self.out_dir}/{experiment_name}/{i}_{j}_{k}.png",
                nrow=len(inversion.final_pred),
            )

            os.makedirs(fakes_path, exist_ok=True)
            for l, image in enumerate(midpoint):
                save_image(
                    image.unsqueeze(0),
                    f"{fakes_path}/{i}_{j}_{k}_{l}.png",
                )

        return fakes_path

    def all_image_pairs(self, dataloader: InvertedDataloader):
        for i, inversion in enumerate(dataloader):
            for j, other_inversion in enumerate(dataloader):
                if i <= j:
                    for k in range(
                        1 if i == j else 0, dataloader.target_dataloader.batch_size
                    ):  # * Skips interpolating images with themselves
                        yield (i, j, k), (inversion, other_inversion)

    def num_image_pairs(self, dataloader: InvertedDataloader):
        n = len(dataloader) * dataloader.batch_size
        return n * (n + 1) // 2
        # 3 batches, 1 with 4 images and 2 with 3  = 10 total images

    def create_artifacts(self, dataloaders: Dict[str, InvertedDataloader]):
        results = self.load_results_from_disk()
        self.make_table(results)

