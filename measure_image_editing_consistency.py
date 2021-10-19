from inversion import *

from training.dataset import ImageFolderDataset
import training.networks as networks

OUT_DIR = "eval/image_editing_consistency"

def measure_image_editing_consistency(
    outdir: str,
    dataloader: Iterable[torch.Tensor],
    alpha: float = 0.5,
):
    losses = []
    inverter = dataloader.inverter

    for i, inversion in enumerate(dataloader):
        target_2_var = inverter.variable_type.sample_random_from(
            inverter.G, batch_size=len(inversion.target)
        )
        
        first_inversion = inversion
        
        with torch.no_grad():
            edit = (target_2_var - first_inversion.final_variable.cuda()) * alpha
            edited_image = (first_inversion.final_variable.cuda() + edit).to_image()

        second_inversion = inversion.rerun

        with torch.no_grad():
            unedited_image = (second_inversion.final_variable.cuda() - edit).to_image()
            losses.append(inverter.criterion(unedited_image, inversion.target))

        save_image(
            torch.cat(
                (
                    target_2_var.to_image(),
                    inversion.target,
                    first_inversion.final_pred.cuda(),
                    edited_image,
                    second_inversion.final_pred.cuda(),
                    unedited_image,
                )
            ),
            f"{outdir}/{i}.png",
            nrow=len(inversion.target),
        )

    return torch.cat(losses)


def run_image_editing_consistency(experiment_name, dataloader):
    fresh_dir(OUT_DIR)
    experiment_dir = OUT_DIR + "/" + experiment_name
    fresh_dir(experiment_dir)

    losses = measure_image_editing_consistency(
        experiment_dir,
        dataloader,
    )

    with open(f"{OUT_DIR}/losses_{experiment_name.lower()}.txt", "w") as file:
        print(losses, file=file) # todo pkl
    with open(f"{OUT_DIR}/image_editing_consistency_{experiment_name.lower()}.txt", "w") as file:
        print(experiment_name, ":", round(losses.mean().item(), 6), file=file)


if __name__ == "__main__":
    results = {}
    G_adain = open_generator("pretrained/adain-normalized.pkl")
    G_adaconv = open_generator("pretrained/adaconv-normalized.pkl")
    for experiment_name, (G, variable_type, learning_rate) in {
        "AdaConvW+": (
            open_generator("pretrained/adaconv-normalized.pkl"),
            WVariable,
            0.1,
        ),
        "AdaINW+": (
            open_generator("pretrained/adain-their-params-003800.pkl"),
            WPlusVariable,
            0.1,
        ),
        "AdaConvW": (
            open_generator("pretrained/adaconv-normalized.pkl"),
            WVariable,
            0.1,
        ),
        "AdaINW": (open_generator("pretrained/adain-normalized.pkl"), WVariable, 0.1),
        "AdaConvZ": (
            open_generator("pretrained/adaconv-gamma-20-003800.pkl"),
            ZVariable,
            0.03,
        ),
        "AdaInZ": (
            open_generator("pretrained/adaconv-gamma-20-003800.pkl"),
            ZVariable,
            0.03,
        ),
        "AdaConvZ+": (
            open_generator("pretrained/adaconv-gamma-20-003800.pkl"),
            ZVariable,
            0.03,
        ),
        "AdaInZ+": (
            open_generator("pretrained/adaconv-gamma-20-003800.pkl"),
            ZVariable,
            0.03,
        ),
    }.items():
        experiment_dir = OUT_DIR + "/" + experiment_name
        os.makedirs(experiment_dir)
        losses = measure_image_editing_consistency(
            experiment_dir,
            RealDataloader(
                "datasets/afhq2_cat128_test.zip", batch_size=4, num_images=8
            ),
            Inverter(
                G,
                num_steps=500,
                learning_rate=learning_rate,
                variable_type=variable_type,
            ),
        )
