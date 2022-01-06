from ..prelude import *

from .eval import *

sys.path.append("vendor/FID_IS_infinity")

from pytorch_fid.fid_score import calculate_fid_given_paths
# from score_infinity import calculate_FID_infinity_path


class EvalInterpolationQuality(Evaluation):
    name: ClassVar[str] = "Interpolation Quality"

    table_stat: str = "FID"

    @torch.no_grad()
    def eval(self, experiment_name: str, dataloader: InvertedDataloader) -> Result:
        for i, inversion in enumerate(dataloader):
            for j, other_inversion in enumerate(dataloader):
                if i <= j:
                    for k in range(
                        1 if i == j else 0, dataloader.target_dataloader.batch_size
                    ): # * Doesn't interpolate images with themselves
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
                        fake_path = f"{self.out_dir}/{experiment_name}/fakes"
                        os.makedirs(fake_path, exist_ok=True)
                        for l, image in enumerate(midpoint):
                            save_image(
                                image.unsqueeze(0),
                                f"{fake_path}/{i}_{j}_{k}_{l}.png",
                            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_avail_cpus = len(os.sched_getaffinity(0))
        num_workers = min(num_avail_cpus, 8)
        # breakpoint()
        # FID_infinity = calculate_FID_infinity_path('datasets/afhq2_cat256_stats.npz', "evaevaluation-runsl/2021-11-25_21:22:21/interpolation_quality/AdaConv/W+/fakes", 32, min_fake=250)

        return {
            "FID": torch.tensor(
                calculate_fid_given_paths(
                    [dataloader.target_dataloader.fid_data_path, fake_path],
                    50,
                    device,
                    2048,
                    num_workers,
                )
            )
        }

    def create_artifacts(self, dataloaders: Dict[str, InvertedDataloader]):
        results = self.load_results_from_disk()
        self.make_table(results)
