from ..prelude import *

from .eval_image_editing_consistency import *
from .eval_interpolation_determinism import *
from .eval_reconstruction_quality import *
from .eval_reconstruction_realism import *
from .eval_interpolation_realism import *


def run_eval(
    target_dataloader: RealDataloader,
    experiments: Dict[str, Tuple[networks.Generator, Type[Variable]]],
    evaluations: List[Evaluation],
    num_steps: int,
    label: Optional[str] = None,
    perform_dry_run: bool = True,
    **kwargs,
) -> None:
    assert len(evaluations) > 0

    for is_dry_run in [True, False] if perform_dry_run else [False]:
        if is_dry_run:
            print("-------------------")
            print("\n👉 Performing dry run...")
            print()
        else:
            print("\n-------------------")
            print("\n👉 Performing full run...")
            print()

        timestamp = create_eval_directory(label, is_dry_run)
        target_dataloader = (
            target_dataloader
            if not is_dry_run
            else target_dataloader.subset(2 * target_dataloader.batch_size)
        )
        
        target_dataloader.serialize(f"evaluation-runs/{timestamp}")

        dataloaders = {}
        for experiment_name, (G_or_E, variable_type) in experiments.items():
            print(experiment_name, flush=True)

            dataloaders[experiment_name] = InvertedDataloader(
                target_dataloader,
                Inverter(
                    G_or_E,
                    num_steps=num_steps if not is_dry_run else 1,
                    variable=variable_type,
                    seed=target_dataloader.seed,
                    **kwargs,
                ),
            )

            for evaluation in evaluations:
                print("\n🧮", evaluation.name, flush=True)
                evaluation(timestamp)(dataloaders)

    create_artifacts(timestamp, evaluations)
    
    return timestamp


def create_eval_directory(label: Optional[str], dry_run: bool) -> str:
    os.makedirs("evaluation-runs", exist_ok=True)

    if dry_run:
        timestamp = "_dry_run"
    else:
        timestamp = str(datetime.now()).split(".")[0].replace(" ", "_")
        timestamp = label + "/" + timestamp if label is not None else timestamp

    fresh_dir(f"evaluation-runs/{timestamp}")

    shutil.copyfile(sys.argv[0], f"evaluation-runs/{timestamp}/config.txt")

    return timestamp


def create_artifacts(
    timestamp: str,
    evaluations: List[Type[Evaluation]],
) -> None:
    fresh_dir(f"evaluation-runs/{timestamp}/artifacts")

    target_dataloader = RealDataloader.deserialize(f"evaluation-runs/{timestamp}")

    for evaluation_type in evaluations:
        evaluation = evaluation_type(timestamp)
        fresh_dir(evaluation.artifacts_dir)
        evaluation.create_artifacts(target_dataloader)

    join_evaluation_tables(timestamp)


def regenerate_artifacts(path: str, evaluations: List[Type[Evaluation]]):
    print(f"Regenerating artifacts from {path}...")
    create_artifacts("/".join(path.split("/")[1:]), evaluations)


def join_evaluation_tables(timestamp: str) -> None:
    os.chdir(f"evaluation-runs/{timestamp}/artifacts")
    print("\n-------------------")
    print("\n👌\n")

    with open(f"full_table.txt", "w") as out_file:
        for path in os.listdir(f"."):
            if os.path.isdir(f"{path}"):
                with open(f"{path}/table.txt", "r") as in_file:
                    data = in_file.read()
                    print(data)
                    print(data, file=out_file)

    print("-------------------")

    os.chdir("../..")
