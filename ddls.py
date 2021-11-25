from inversion import *

from spectral import *
from training.networks import Discriminator

G_PATH = f"pretrained/nombstd-adain.pkl"

if __name__ == "__main__":
    G, D = open_generator_and_discriminator(G_PATH)

    criterion = 100 * NullCriterion()
    target = open_target(
        G,
        "datasets/afhq2/test/cat/flickr_cat_000176.png",
        "datasets/afhq2/test/cat/flickr_cat_000236.png",
    )

    inverter = Inverter(
        G,
        3000,
        variable_type=ZVariableConstrainToTypicalSet,
        create_optimizer=lambda params: SGDWithNoise(
            params, lr=0.1, noise_amount=1.0, noise_sparsness=0.0
        ),
        create_schedule=lambda optimizer: lr_scheduler.LambdaLR(
            optimizer, lambda epoch: min(1.0, epoch / 100.0) #* 0.998 ** epoch
        ),
        criterion=criterion,
        snapshot_frequency=10,
        seed=7,
    )

    for inversion, do_snapshot in tqdm.tqdm(
        inverter.all_inversion_steps(target),
    ):
        if do_snapshot:
            save_image(
                torch.cat(
                    (
                        inversion.final_pred,
                        inversion.final_variable.to_image(truncation=0.5),
                    )
                ),
                "out/ddls.png",
                nrow=len(target),
            )
