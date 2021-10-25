from .eval_image_editing_consistency import *
from .eval_interpolation_determinism import *
from .eval_reconstruction_quality import *


def run_eval(
    label: str,
    target_dataloader: RealDataloader,
    variable_types: List[Type[Variable]],
    methods: Dict[str, networks.Generator],
    evaluations: List[Evaluation],
    num_steps: int,
) -> None:
    for dry_run in [True, False] if num_steps > 1 else [True]:
        timestamp = create_eval_directory(label, dry_run)
        target_dataloader = (
            target_dataloader
            if not dry_run
            else target_dataloader.subset(2 * target_dataloader.batch_size)
        )

        for variable_type in variable_types:
            dataloaders = {}
            for method_name, G in methods.items():
                experiment_name = method_name + "/" + variable_type.space_name
                print(experiment_name, flush=True)

                dataloaders[experiment_name] = InvertedDataloader(
                    target_dataloader,
                    Inverter(
                        G,
                        num_steps=num_steps if not dry_run else 1,
                        variable_type=variable_type,
                        seed=target_dataloader.seed,
                    ),
                )

            for evaluation in evaluations:
                evaluation(timestamp)(dataloaders)
    
    return timestamp


def create_eval_directory(label: str, dry_run: bool) -> str:
    os.makedirs("eval", exist_ok=True)

    prefix = "" if label == "" else label + "_"
    if dry_run:
        timestamp = prefix + "dry_run"
    else:
        timestamp = prefix + str(datetime.now()).split(".")[0].replace(" ", "_")

    fresh_dir(f"eval/{timestamp}")
    shutil.copyfile(__file__, f"eval/{timestamp}/config.txt")

    return timestamp


def create_artifacts(
    timestamp: str, target_dataloader: RealDataloader, evaluations: List[Evaluation]
) -> None:
    for evaluation in evaluations:
        evaluation(timestamp).create_artifacts(target_dataloader)
    join_evaluation_tables(timestamp)


def join_evaluation_tables(timestamp: str) -> None:
    os.chdir(f"eval/{timestamp}")

    with open(f"full_table.txt", "w") as out_file:
        for path in os.listdir(f"."):
            if os.path.isdir(f"{path}"):
                with open(f"{path}/table.txt", "r") as in_file:
                    data = in_file.read()
                    print(data)
                    print(data, file=out_file)

    os.chdir("../..")
