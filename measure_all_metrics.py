from inversion import *

from measure_reconstruction_quality import run_reconstruction_quality
from measure_image_editing_consistency import run_image_editing_consistency
from measure_interpolation_determinism import run_interpolation_determinism

# todo extract make_table to a shared function
# todo extract loop in each metric to a shared function
# todo add a error histogram
# todo save multiple folders, one per eval run
# todo do a cross product for both consistency losses
# todo two pass: generate data, then plots and tables
# todo double check that it is deterministic for a given seed, & stable for another (low variance)
# todo run fully to check "slow" trainings
# todo move eveyrhting inside a folder


def run(
    dataloader: RealDataloader,
    variable_types: List[Type[Variable]],
    methods: Dict[str, networks.Generator],
    num_steps: int,
):
    fresh_dir("eval")
    shutil.copyfile(__file__, f"eval/config.txt")

    for variable_type in variable_types:
        dataloaders = {}
        for method_name, G in methods.items():
            experiment_name = method_name + "/" + variable_type.space_name
            print(experiment_name, flush=True)

            dataloaders[experiment_name] = InvertedDataloader(
                dataloader,
                Inverter(
                    G,
                    num_steps=num_steps,
                    variable_type=variable_type,
                    seed=dataloader.seed,
                ),
            )

        run_reconstruction_quality(dataloaders)
        run_interpolation_determinism(dataloaders)
        run_image_editing_consistency(dataloaders)

    join_tables()


def join_tables():
    with open(f"eval/full_table.txt", "w") as out_file:
        for path in os.listdir("eval"):
            if os.path.isdir(f"eval/{path}"):
                with open(f"eval/{path}/table.txt", "r") as in_file:
                    data = in_file.read()
                    print(data)
                    print(data, file=out_file)


if __name__ == "__main__":
    run(
        dataloader=RealDataloader(
            "datasets/afhq2_cat128_test.zip",
            batch_size=4,
            num_images=12,
        ),
        variable_types=[ZVariable, ZPlusVariable, WVariable, WPlusVariable],
        num_steps=2,
        methods={
            "AdaConv": open_generator("pretrained/adaconv-gamma-20-003800.pkl"),
            "AdaConvSlow": open_generator("pretrained/adaconv-slowdown-all.pkl"),
        },
    )
