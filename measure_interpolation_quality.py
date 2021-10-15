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
                result = z1.to_image(), z2.to_image(), Interpolation.mix_between(z1.to_W(), z2.to_W())
            yield result


def measure_interpolation_quality(
    dir: str,
    dataloader: FakeMixDataloader,
    inverter: Inverter,
):
    os.makedirs(dir)

    losses = []

    for i, (target_A, target_B, target_mix) in enumerate(iter(dataloader)):
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
                    target_mix,
                    mix,
                ]
            ),
            f"{dir}/mix_{i}.png",
            nrow=2
        )

    return losses


def make_plot(path: str, losses: List[float]):
    plt.title("Interpolation Quality (distance to expr)")
    plt.plot(torch.stack(losses).cpu())
    plt.savefig(path)


if __name__ == "__main__":
    fresh_dir(OUTDIR)
    torch.manual_seed(0)

    for method_name, make_generator in {
        "AdaIN": lambda: open_generator(latest_snapshot("cfg_auto_large_res_adain")),
        "AdaConv": lambda: open_generator(
            latest_snapshot("cfg_auto_large_res_adaconv")
        ),
    }.items():
        G = make_generator()
        dataloader = FakeMixDataloader(G, batch_size=1, num_images=10)
        losses = measure_interpolation_quality(
            f"{OUTDIR}/{method_name.lower()}",
            dataloader,
            Inverter(
                G,
                num_steps=300,
                learning_rate=0.3,
                beta_1=0.0,
                beta_2=0.0,
                variable_type=WPlusVariable,
            ),
        )
        breakpoint()
        make_plot(f"{OUTDIR}/{method_name}", losses)
