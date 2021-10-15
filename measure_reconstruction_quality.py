from inversion import *


# todo fix hack G init
# todo run on fake data
# todo batching
# todo hyperparam search combined plot
# todo serialize data first, then reload it for producing plots

OUTDIR = "out/reconstruction_quality"


def measure_reconstruction_quality(
    dir: str, inverter: Inverter, dataloader: Iterable[torch.Tensor]
):
    os.makedirs(dir)

    all_losses = []

    for i, target in enumerate(dataloader):
        inversion = inverter(target, out_path=f"{dir}/{i}.png")
        all_losses.append(inversion.losses)

    return all_losses


def make_table(all_losses_by_method: Dict[str, List[List[float]]]):
    for method_name, all_losses in all_losses_by_method.items():
        all_final_losses = torch.tensor([losses[-1] for losses in all_losses])
        print(
            f"{method_name}: {all_final_losses.mean().item():.04f} +- {all_final_losses.std().item():.04f}"
        )


def make_plot(
    path: Optional[str],
    all_losses_by_method: Dict[str, List[List[float]]],
    description: str = "",
):
    plt.figure()
    plt.title(f"[min, Q1, median, Q3, max] {description}")
    plt.suptitle("Reconstruction Error Bounds & Quantiles per Optimization Step")
    method_names = []
    cmap = plt.get_cmap("tab10")
    for i, (method_name, all_losses) in enumerate(all_losses_by_method.items()):
        method_names.append(method_name)
        all_losses = torch.stack([torch.tensor(losses) for losses in all_losses])
        low = all_losses.amin(dim=0)
        q1 = all_losses.quantile(0.25, dim=0)
        medians = all_losses.median(dim=0).values
        q3 = all_losses.quantile(0.75, dim=0)
        high = all_losses.amax(dim=0)
        assert all(
            data.shape == all_losses[0].shape for data in [low, q1, medians, q3, high]
        )
        plt.plot(medians, color=cmap(i))
        plt.fill_between(range(all_losses.shape[1]), q1, q3, alpha=0.3, color=cmap(i))
        plt.fill_between(
            range(all_losses.shape[1]), low, high, alpha=0.2, color=cmap(i)
        )
    plt.legend(method_names)
    plt.xlabel("Optimization Step")
    plt.ylabel("Reconstruction Error")
    plt.ylim(0)
    plt.savefig(path)


def run(
    label: Optional[str],
    inverters: Dict[str, Inverter],
    loaders: Dict[str, InversionDataloader]
):
    all_losses_by_method = {}

    for method_name, inverter in inverters.items():
        loader = loaders[method_name]
        all_losses = measure_reconstruction_quality(
            f"{OUTDIR}/{loader.name}{'_' if label is not None else ''}{label}/{method_name}",
            inverter,
            loader,
        )
        all_losses_by_method[method_name] = all_losses

    make_table(all_losses_by_method)
    make_plot(
        f"{OUTDIR}/plot{'_' if label is not None else ''}{label}.png",
        all_losses_by_method,
        f"over {loader.num_images} {loader.name} images ({inverter.variable_type.space_name})",
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    fresh_dir(OUTDIR)

    methods = {
        "AdaConv": 
            Inverter(
                open_generator(latest_snapshot("cfg_auto_large_res_adaconv")), 
                num_steps=300,
                learning_rate=0.1,
                beta_1=0.0,
                beta_2=0.0,
                variable_type=WPlusVariable,
            ),
        "AdaIN": Inverter(
                open_generator(latest_snapshot("cfg_auto_large_res_adain")),
                num_steps=300,
                learning_rate=0.1,
                beta_1=0.0,
                beta_2=0.0,
                variable_type=WPlusVariable,
            ),
    }
    run(
        None,
        methods,
        {
            "AdaConv": FakeDataloader(methods["AdaConv"].G, batch_size=1, num_images=20),
            "AdaIN": FakeDataloader(methods["AdaIN"].G, batch_size=1, num_images=20),
        },
    )
