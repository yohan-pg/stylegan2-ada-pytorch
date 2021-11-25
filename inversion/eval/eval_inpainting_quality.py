from ..prelude import *

from .eval import *


class EvalInpaintingQuality(Evaluation):
    name: ClassVar[str] = "Inpainting Quality"

    @torch.no_grad()
    def eval(self, experiment_name: str, dataloader: InvertedDataloader) -> Result:
        inverter = None 

        for i, reference_inversion in enumerate(dataloader):
            inversion = inverter(reference_inversion.target)
            

        breakpoint()

        return {}

        # return { "FID": torch.cat(losses) }

    def create_artifacts(self, dataloaders: Dict[str, InvertedDataloader]):
        results = self.load_results_from_disk()
        self.make_table(results)
        self.make_histogram(results)
