from inversion import *


OUT_DIR = "eval/reconstruction_quality"


def measure_reconstruction_quality(dir: str, dataloader: InvertedDataloader):
    os.makedirs(dir)

    all_losses = []

    for _, inversion in enumerate(dataloader):
        all_losses.append(torch.stack(inversion.losses, dim=1))

    return torch.cat(all_losses)


def make_table(experiment_name: str, losses: Dict[str, torch.Tensor]):
    with open(f"{OUT_DIR}/table.txt", "w") as f:
        all_final_losses = losses[:, -1]
        print(
            f"{experiment_name}: {all_final_losses.mean().item():.04f} +- {all_final_losses.std().item():.04f}",
            file=f,
        )


@torch.no_grad()
def make_plot(
    experiment_name: str,
    all_losses: Dict[str, torch.Tensor],
    description: str = "",
):
    plt.figure()
    plt.title(f"[min, Q1, median, Q3, max] {description}")
    plt.suptitle("Reconstruction Error Bounds & Quantiles per Optimization Step")
    # cmap = plt.get_cmap("tab10")

    low = all_losses.amin(dim=0)
    q1 = all_losses.quantile(0.25, dim=0)
    medians = all_losses.median(dim=0).values
    q3 = all_losses.quantile(0.75, dim=0)
    high = all_losses.amax(dim=0)
    assert all(
        data.shape == all_losses[0].shape for data in [low, q1, medians, q3, high]
    )
    # plt.legend(experiment_names)
    # plt.plot(medians.cpu(), color=cmap(i))
    plt.fill_between(
        range(all_losses.shape[1]),
        q1.cpu(),
        q3.cpu(),
        alpha=0.3,  # color=cmap(i)
    )
    plt.fill_between(
        range(all_losses.shape[1]),
        low.cpu(),
        high.cpu(),
        alpha=0.2,  # color=cmap(i)
    )
    plt.ylim(0.0, 0.6)

    plt.xlabel("Optimization Step")
    plt.ylabel("Reconstruction Error")
    plt.ylim(0)
    plt.savefig(f"{OUT_DIR}/plot_{experiment_name.lower()}.png")


def run_reconstruction_quality(
    experiment_name: Optional[str], dataloader: InvertedDataloader
):
    fresh_dir(OUT_DIR)

    all_losses = measure_reconstruction_quality(
        f"{OUT_DIR}/{experiment_name}", dataloader
    )

    make_table(experiment_name, all_losses)
    make_plot(
        experiment_name,
        all_losses,
        f"over {dataloader.num_images} images ({dataloader.inverter.variable_type.space_name})",
    )


if __name__ == "__main__":
    fresh_dir(OUT_DIR)

    dataloader = RealDataloader(
        "datasets/afhq2_cat128_test.zip", batch_size=4, num_images=20
    )

    for VARIABLE_TYPE in [ZVariable, WPlusVariable, WVariable, ZPlusVariable]:
        methods = {
            "AdaConv": Inverter(
                open_generator("pretrained/adaconv-gamma-20-003800.pkl"),
                num_steps=300,
                learning_rate=VARIABLE_TYPE.default_lr,
                variable_type=VARIABLE_TYPE,
            ),
            "AdaConvSlow": Inverter(
                open_generator("pretrained/adaconv-slowdown-all.pkl"),
                num_steps=300,
                learning_rate=VARIABLE_TYPE.default_lr,
                variable_type=VARIABLE_TYPE,
            ),
        }

        run_reconstruction_quality(
            str(VARIABLE_TYPE.__name__.lower() + str(VARIABLE_TYPE.default_lr)),
            methods,
            dataloader,
        )
