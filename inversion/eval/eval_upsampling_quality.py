from ..prelude import *

from .eval import *


class EvalUpsamplingQuality(Evaluation):
    name: ClassVar[str] = "Upsampling Quality"

    @torch.no_grad()
    def eval(self, experiment_name: str, dataloader: InvertedDataloader) -> Result:
        for i, inversion in enumerate(dataloader):
            for j, other_inversion in enumerate(dataloader):
                if i < j:
                    midpoint = inversion.final_variable.interpolate(
                            other_inversion.final_variable, 0.5
                        ).to_image()
                    save_image(
                        midpoint,
                        f"{self.out_dir}/{experiment_name}/{i}.png",
                        nrow=len(inversion.final_pred),
                    )

        breakpoint()

        return {}

        # return { "FID": torch.cat(losses) }

    def create_artifacts(self, dataloaders: Dict[str, InvertedDataloader]):
        results = self.load_results_from_disk()
        self.make_table(results)
        self.make_histogram(results)
