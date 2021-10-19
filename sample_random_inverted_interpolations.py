from inversion import *

from training.dataset import ImageFolderDataset
import training.networks as networks
import itertools 

OUTDIR = "out/inverted_interpolation"

SAME_SEED = True


def sample_random_inverted_interpolations(
    dir: str,
    dataloader: RealDataloader,
    inverter: Inverter,
):
    os.makedirs(dir)

    losses = []

    for i, (target_A, target_B) in enumerate(grouper(dataloader, 2)):
        if SAME_SEED:
            torch.manual_seed(0)
        inversion_A = inverter(target_A)
        if SAME_SEED:
            torch.manual_seed(0)
        inversion_B = inverter(target_B)
        Interpolation.from_variables(
            inversion_A.final_variable, inversion_B.final_variable
        ).save(f"{dir}/{i}.png", target_A, target_B)

    return losses


if __name__ == "__main__":
    fresh_dir(OUTDIR)

    for method_name, G in {
        "AdaConv": open_generator("pretrained/adaconv-gamma-20-003800.pkl"),
        # "AdaIN": open_generator("pretrained/adain-their-params-003800.pkl")
    }.items():
        for variable_type in [ZVariableConstrainToTypicalSetAllVecs, ZVariable]:
            torch.manual_seed(0)
            dataloader = RealDataloader("datasets/afhq2_cat128_test.zip", batch_size=4, num_images=8)
            losses = sample_random_inverted_interpolations(
                f"{OUTDIR}/{method_name.lower()}_{variable_type.__name__.lower()}",
                dataloader,
                Inverter(
                    G,
                    num_steps=300,
                    learning_rate=0.03,
                    variable_type=variable_type,
                ),
            )
