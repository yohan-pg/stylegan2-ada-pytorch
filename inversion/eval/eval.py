from ..io import *
from ..prelude import *
from ..inverter import *

from datetime import datetime

Result = Dict[str, torch.Tensor]
Results = Dict[str, Result]


class Evaluation:
    name: ClassVar[str]

    @property
    def out_dir(self):
        return f"eval/{self.timestamp}/{self.name.lower().replace(' ', '_')}"

    def __init__(self, timestamp: datetime):
        self.timestamp = timestamp

    def eval(self, experiment_name: str, dataloader: InvertedDataloader) -> Result:
        raise NotImplementedError

    def create_artifacts(
        self, target_dataloader: RealDataloader, results: Results
    ):
        raise NotImplementedError

    def run(self, dataloaders: Dict[str, InvertedDataloader]) -> Results:
        target_dataloader = next(iter(dataloaders.values())).target_dataloader
        for dataloader in dataloaders.values():
            assert dataloader.target_dataloader is target_dataloader

        results = {}

        for experiment_name, dataloader in dataloaders.items():
            os.makedirs(f"{self.out_dir}/{experiment_name}", exist_ok=True)

            results[experiment_name] = self.eval(experiment_name, dataloader)

            with open(f"{self.out_dir}/{experiment_name}/result.pkl", "wb") as file:
                pickle.dump(results[experiment_name], file)

        return results

    def load_results_from_disk(self) -> Results:
        for directory in os.listdir(self.out_dir):
            pass

    __call__ = run

    def make_table(self, results) -> None:
        file_path = f"{self.out_dir}/table.txt"
        
        if not os.path.exists(file_path):
            with open(file_path, "w") as file:
                print(f"## {self.name}", file=file)
        
        with open(file_path, "a") as file:
            max_len_name = max(map(len,results.keys()))
            for experiment_name, result in results.items():
                print(
                    f"{experiment_name.ljust(max_len_name + 1, ' ')}: {result['losses'].mean().item():.04f} +- {result['losses'].std().item():.04f}",
                    file=file,
                )

    def make_histogram(self, results):
        plt.figure()
        
        names = []
        losses = []
        for experiment_name, result in results.items():
            losses.append(result["losses"].cpu().numpy())
            names.append(experiment_name)
        
        plt.hist(losses)
        plt.xlim(left=0)
        plt.title(f"{self.name} Loss Distribution")
        plt.legend(names)
        plt.savefig(f"{self.out_dir}/histogram.png")
