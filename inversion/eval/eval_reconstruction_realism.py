from ..prelude import *

from .eval import *

from .fid import *


class EvalReconstructionRealism(Evaluation):
    name: ClassVar[str] = "Reconstruction Realism"

    table_stat: str = "FID"

    @torch.no_grad()
    def eval(self, experiment_name: str, dataloader: InvertedDataloader) -> Result:
        path = self.save_images(experiment_name, dataloader)
        return {"FID": compute_fid(dataloader, path)}

    def save_images(self, experiment_name: str, dataloader: InvertedDataloader) -> str:
        print("Saving Reconstruction images...")

        path = f"{self.out_dir}/{experiment_name}"
        for (i, j), image in self.all_images(dataloader):
            save_image(image.unsqueeze(0), f"{path}/{i}_{j}.png")

        return path

    def all_images(self, dataloader: InversionDataloader):
        for i, inversion in enumerate(dataloader):
            for j, image in enumerate(inversion.final_pred):
                yield (i, j), image

    def num_images(self, dataloder: InversionDataloader):
        return len(dataloder) * dataloder.batch_size

    def create_artifacts(self, dataloaders: Dict[str, InvertedDataloader]):
        results = self.load_results_from_disk()
        self.make_table(results)
