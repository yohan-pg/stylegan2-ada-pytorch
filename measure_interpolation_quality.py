from inversion import *

from training.dataset import ImageFolderDataset
import training.networks as networks

OUTDIR = "out/interpolation_quality"


class FakeMixDataloader(FakeDataloader):
    def __iter__(self):
        for _ in range(self.num_images):
            with torch.no_grad():
                z1 = ZVariable.sample_from(self.G, self.batch_size)
                z2 = ZVariable.sample_from(self.G, self.batch_size)
                yield z1.to_image(), z2.to_image(), Interpolation.mix_between(z1, z2)


def measure_interpolation_quality(
    dir: str,
    inverter: Inverter,
    dataloader: FakeDataloader,
):

    losses = []

    for i, (target_A, target_B, target_mix) in enumerate(dataloader):
        inversion_A = inverter(target_A)
        inversion_B = inverter(target_B)
        mix = Interpolation.mix_between(
            inversion_A.final_variable, inversion_B.final_variable
        )
        losses.append(
            inverter.criterion(
                mix,
                target_mix,
            )
        )
        save_image(
            torch.cat(
                [
                    target_A,
                    inversion_A.final_pred,
                    target_B,
                    inversion_B.final_pred,
                    mix,
                    target_mix,
                ]
            ),
            f"{dir}/mix_{i}.png",
        )

    return losses


def make_plot(path: str, losses: List[float]):
    plt.title("Interpolation Quality (distance to expr)")
    plt.plot(losses)
    plt.savefig(path)


def run(method_name: str, dataloader: FakeDataloader, inverter: Inverter):
    pass


if __name__ == "__main__":
    fresh_dir(OUTDIR)
    torch.manual_seed(0)

    for method_name, make_generator in {
        "AdaConv": lambda: open_generator(
            latest_snapshot("cfg_auto_large_res_adaconv")
        ),
        "AdaIN": lambda: open_generator(latest_snapshot("cfg_auto_large_res_adain")),
    }.items():
        G = make_generator()
        loader = FakeDataloader(G)
        run(
            method_name,
            Inverter(
                None,  # * Populated within the loop; a bit of a hack
                num_steps=300,
                learning_rate=0.3,
                beta_1=0.0,
                beta_2=0.0,
                variable_type=WPlusVariable,
            ),
            loader,
        )
