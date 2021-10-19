from inversion import *

from training.dataset import ImageFolderDataset
import training.networks as networks

OUT_DIR = "eval/interpolation_determinism"

def measure_interpolation_determinism(
    outdir: str,
    dataloader: InvertedDataloader,
    alpha: float = 0.5,
):
    losses = []
    
    for i, (inversion, interpolation_inversion) in tqdm.tqdm(
        enumerate(grouper(dataloader, 2))
    ):
        if interpolation_inversion is None:
            break
        
        first_inversion = inversion
        second_inversion = inversion.rerun

        with torch.no_grad():
            for j, var in enumerate(
                interpolation_inversion.final_variable.cuda().to_W().split_into_individual_variables()
            ):
                first_interpolation = (
                    first_inversion.final_variable.cuda().to_W()
                    .interpolate(var, alpha)
                    .to_image()
                )
                second_interpolation = (
                    second_inversion.final_variable.cuda().to_W()
                    .interpolate(var, alpha)
                    .to_image()
                )

                losses.append(
                    dataloader.inverter.criterion(first_interpolation, second_interpolation)
                )

                def interleave(a, b):
                    return torch.stack((a, b), dim=1).flatten(start_dim=0, end_dim=1)

                save_image(
                    torch.cat(
                        [
                            interleave(inversion.target, inversion.target),
                            interleave(
                                first_inversion.final_pred.cuda(),
                                second_inversion.final_pred.cuda(),
                            ),
                            interleave(first_interpolation, second_interpolation),
                            torch.cat([var.to_image()] * (2 * len(inversion.target))),
                            torch.cat(
                                [interpolation_inversion.target[j : j + 1]] * (2 * len(inversion.target))
                            ),
                        ]
                    ),
                    f"{outdir}/{i}_{j}.png",
                    nrow=len(inversion.target) * 2,
                )

    return torch.cat(losses)


def make_histogram(losses):
    plt.figure()
    plt.hist(losses.cpu(), torch.ones(10).cpu())
    plt.savefig(f"{OUT_DIR}/hist.png")


def run_interpolation_determinism(
    method_name: str, dataloader: RealDataloader
):
    fresh_dir(OUT_DIR)
    method_dir = OUT_DIR + "/" + method_name
    fresh_dir(method_dir)

    results = {}
    losses = measure_interpolation_determinism(
        method_dir,
        dataloader,
    )
    with open(f"{method_dir}/losses_{method_name}.py", "w") as file:
        print(losses, file=file)
    results[method_name] = losses

    return results


if __name__ == "__main__":
    adaconv = open_generator("pretrained/adaconv-gamma-20-003800.pkl")
    adaconv_slowdown = open_generator("pretrained/adaconv-slowdown-all.pkl")
    dataloader = RealDataloader(
        "datasets/afhq2_cat128_test.zip", batch_size=4, num_images=20
    )

    for method_name, (G, variable_type, num_steps) in {
        "AdaConvW": (adaconv, WVariable, 200),
        "AdaconvSlowW": (adaconv_slowdown, WVariable, 200),
        "AdaConvW+": (adaconv, WPlusVariable, 200),
        "AdaconvSlowW+": (adaconv_slowdown, WPlusVariable, 200),
    }.items():
        run_interpolation_determinism(method_name, dataloader)
