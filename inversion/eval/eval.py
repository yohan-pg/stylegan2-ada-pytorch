from ..io import *
from ..prelude import *
from ..inverter import *

from datetime import datetime
import glob

Result = Dict[str, torch.Tensor]
Results = Dict[str, Result]


class Evaluation:
    name: ClassVar[str]

    def __init__(self, timestamp: datetime):
        self.timestamp = timestamp
        self.artifacts_dir = f"evaluation-runs/{self.timestamp}/artifacts/{self.name.lower().replace(' ', '_')}"
        self.out_dir = self.artifacts_dir.replace("/artifacts/", "/inversion/")

    def eval(self, experiment_name: str, dataloader: InvertedDataloader) -> Result:
        raise NotImplementedError

    def create_artifacts(self, target_dataloader: RealDataloader):
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
        results = {}

        for result_path in glob.glob(f"{self.out_dir}/**/result.pkl", recursive=True):
            name = "/".join(result_path.replace(self.out_dir + "/", "").split("/")[:-1])
            results[name] = pickle.load(open(result_path, "rb"))

        return results

    __call__ = run

    table_stat: str = "losses"

    def make_table(self, results) -> None:
        file_path = f"{self.artifacts_dir}/table.txt"

        with open(file_path, "w") as file:
            print(f"## {self.name}", file=file)
            max_len_name = max(map(len, results.keys()))
            for experiment_name, result in results.items():
                std = (
                    f"+- {result[self.table_stat].std().item():.04f}"
                    if result[self.table_stat].ndim > 1
                    else ""
                )
                print(
                    f"{experiment_name.ljust(max_len_name + 1, ' ')}: {result[self.table_stat].mean().item():.04f} {std}",
                    file=file,
                )

    def make_histogram(self, results):
        for group_name in set([key.split("/")[1] for key in results.keys()]):
            plt.figure()

            names = []
            losses = []
            for experiment_name, result in results.items():
                if experiment_name.split("/")[1] == group_name:
                    losses.append(result[self.table_stat].cpu().numpy())
                    names.append(experiment_name)

            plt.hist(losses)
            plt.xlim(left=0)
            plt.title(f"{self.name} Loss Distribution")
            plt.legend(names)
            name = f"{self.artifacts_dir}/histogram_{group_name}"
            plt.savefig(f"{name}.svg")
            plt.savefig(f"{name}.png")

    @torch.no_grad()
    def make_plot(
        self,
        target_dataloader: RealDataloader,
        results: Results,
    ) -> None:
        description = f"over {target_dataloader.max_images} images"

        for group_name in set([key.split("/")[1] for key in results.keys()]):
            plt.figure()
            plt.title(f"[min, Q1, median, Q3, max] {description}")
            plt.suptitle(
                "Reconstruction Error Bounds & Quantiles per Optimization Step"
            )

            cmap = plt.get_cmap("tab10")

            names = []
            for i, (experiment_name, result) in enumerate(
                [
                    (experiment_name, result)
                    for (experiment_name, result) in results.items()
                    if experiment_name.split("/")[1] == group_name
                ]
            ):
                all_losses = result["losses_per_step"]
                names.append(experiment_name)
                low = all_losses.amin(dim=0)
                q1 = all_losses.quantile(0.25, dim=0)
                medians = all_losses.median(dim=0).values
                q3 = all_losses.quantile(0.75, dim=0)
                high = all_losses.amax(dim=0)
                assert all(
                    data.shape == all_losses[0].shape
                    for data in [low, q1, medians, q3, high]
                )

                plt.plot(medians.cpu(), color=cmap(i))
                plt.fill_between(
                    range(all_losses.shape[1]),
                    q1.cpu(),
                    q3.cpu(),
                    alpha=0.3,
                    color=cmap(i),
                )
                plt.fill_between(
                    range(all_losses.shape[1]),
                    low.cpu(),
                    high.cpu(),
                    alpha=0.2,
                    color=cmap(i),
                )
                plt.ylim(0.0, 0.6)

            plt.legend(names)

            plt.xlabel("Optimization Step")
            plt.ylabel("Reconstruction Error")
            plt.ylim(0)
            name = f"{self.artifacts_dir}/plot_{group_name}"
            plt.savefig(f"{name}.svg")
            plt.savefig(f"{name}.png")
