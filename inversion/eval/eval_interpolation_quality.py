from ..prelude import *

from .eval import *

from pytorch_fid.fid_score import calculate_fid_given_paths


class EvalInterpolationQuality(Evaluation):
    name: ClassVar[str] = "Interpolation Quality"

    #!!!!!!! this appears to be nondeterminsitic

    @torch.no_grad()
    def eval(self, experiment_name: str, dataloader: InvertedDataloader) -> Result:
        for i, inversion in enumerate(dataloader):
            for j, other_inversion in enumerate(dataloader):
                if i <= j:
                    for k in range(
                        1 if i == j else 0, dataloader.target_dataloader.batch_size
                    ):
                        midpoint = (
                            inversion.final_variable.roll(k)
                            .interpolate(other_inversion.final_variable, 0.5)
                            .to_image()
                        )
                        save_image(
                            torch.cat(
                                [
                                    inversion.target,
                                    inversion.final_pred.roll(k, dims=[0]),
                                    midpoint,
                                    other_inversion.final_pred,
                                ]
                            ),
                            f"{self.out_dir}/{experiment_name}/{i}_{j}_{k}.png",
                            nrow=len(inversion.final_pred),
                        )
                        fake_path = f"{self.out_dir}/{experiment_name}/fakes"
                        os.makedirs(fake_path, exist_ok=True)
                        for l, (image, target) in enumerate(
                            zip(midpoint, inversion.target)
                        ):
                            save_image(
                                image.unsqueeze(0),
                                f"{fake_path}/{i}_{j}_{k}_{l}.png",
                            )

        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
        num_avail_cpus = len(os.sched_getaffinity(0))
        num_workers = min(num_avail_cpus, 8)
        breakpoint()
        return {
            "FID": calculate_fid_given_paths(
                ["datasets/afhq2_cat256", fake_path], #!!!
                50,
                device,
                2048,
                num_workers,
            )
        }

    def create_artifacts(self, dataloaders: Dict[str, InvertedDataloader]):
        results = self.load_results_from_disk()
        self.make_table(results)
        self.make_histogram(results)
