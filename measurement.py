from inversion import *

LossesPerStep = torch.Tensor
Results = Dict[str, LossesPerStep]


class Measurement:
    name: ClassVar[str]
    our_dir: str

    def measure(self, dataloader: InvertedDataloader) -> LossesPerStep:
        raise NotImplementedError

    def create_artifacts(self, results: Results):
        raise NotImplementedError

    def run(self, dataloaders: Dict[str, InvertedDataloader]) -> Results:
        results = {}

        for experiment_name, dataloader in dataloaders.items():
            os.makedirs(f"{self.out_dir}/{experiment_name}", exist_ok=True)

            results[experiment_name] = self.measure(dataloader)

            with open(f"{self.out_dir}/{experiment_name}/losses.pkl", "wb") as file:
                pickle.dump(results[experiment_name], file)

        self.create_artifacts(results)

        return results

    __call__ = run

    def make_table(self, results) -> None:
        with open(f"{self.out_dir}/table.txt", "w") as file:
            print("## Reconstruction Quality", file=file)
            for experiment_name, losses in results.items():
                all_final_losses = losses[:, -1]
                print(
                    f"{experiment_name}: {all_final_losses.mean().item():.04f} +- {all_final_losses.std().item():.04f}",
                    file=file,
                )
