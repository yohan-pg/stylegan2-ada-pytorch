from inversion import *

from spectral import *
from training.networks import Discriminator

import torch.optim.lr_scheduler as lr_scheduler

# G_PATH = f"pretrained/adaconv-256-001200.pkl"
# G_PATH = "pretrained/adain-their-params-003800.pkl"
G_PATH = f"pretrained/adaconv-slowdown-all.pkl"


def make_mask(x):
    x = x.clone()
    mask = torch.ones_like(x)
    _, _, H, W = x.shape
    mask[:, :, :, : W // 2] = 0.0
    return mask


#!!! with an acceptance rate of 0, we should not change at all
#!!! transition density has the wrong arguments I'm pretty sure
#!!! we should be able to degenerate into MCMC with step_size = 0 (if we seperate the noise from LR)
#!!! we should be able to degeenrate into gradient decent iwth noise level = 0

if __name__ == "__main__":
    G = open_generator(G_PATH)

    target = open_target(
        G,
        # "./datasets/afhq2/train/cat/flickr_cat_000018.png",
        # "datasets/samples/cats/00000/img00000014.png",
        # "./datasets/afhq2/train/cat/flickr_cat_000007.png",
        # "datasets/afhq2/test/cat/pixabay_cat_000542.png",
        # "datasets/afhq2/test/cat/pixabay_cat_002694.png"
        # "datasets/afhq2/test/wild/flickr_wild_001251.png"
        # "./datasets/afhq2/train/dog/flickr_dog_000021.png"
        # "./datasets/afhq2/train/dog/flickr_dog_000022.png"
        # A
        "datasets/afhq2/test/cat/flickr_cat_000176.png",
        "datasets/afhq2/test/cat/flickr_cat_000236.png",
        "datasets/afhq2/test/cat/flickr_cat_000368.png",
        "datasets/afhq2/test/cat/pixabay_cat_000117.png",
        # Dogs
        # "datasets/afhq2/test/dog/flickr_dog_000176.png",
        # "datasets/afhq2/test/dog/flickr_dog_000205.png",
        # "datasets/afhq2/test/dog/flickr_dog_000254.png",
        # "datasets/afhq2/test/dog/flickr_dog_000313.png"
        # Cats
        # "datasets/afhq2/train/cat/flickr_cat_000512.png",
        # "datasets/afhq2/train/cat/flickr_cat_000539.png",
        # "datasets/afhq2/train/cat/flickr_cat_000553.png",
        # "datasets/afhq2/train/cat/flickr_cat_000562.png",
        # cats
        # "datasets/afhq2/test/cat/pixabay_cat_002488.png",
        # "datasets/afhq2/test/cat/pixabay_cat_002860.png",
        # "datasets/afhq2/test/cat/pixabay_cat_002905.png",
        # "datasets/afhq2/test/cat/pixabay_cat_002997.png"
    )
    mask = make_mask(target)

    criterion = RightHalfVGGCriterion(mask)
    inverter = Inverter(
        G,
        300,
        variable_type=Z2Variable,
        create_optimizer=lambda params: SGDWithNoise(
            params, lr=250.0, noise_amount=0.005, noise_gamma=1.0
        ),
        create_schedule=lambda optimizer: lr_scheduler.LambdaLR(
            optimizer, lambda epoch: dbg(min(1.0, epoch / 100.0)) * 0.999 ** epoch
        ),
        criterion=criterion,
        snapshot_frequency=10,
        seed=7,
    )

    for inversion in tqdm.tqdm(
        inverter.all_inversion_steps(target * mask), total=len(inverter)
    ):
        result = inversion.ema.to_image()
        save_image(
            torch.cat(
                (
                    target,
                    mask,
                    target * mask,
                    inversion.final_pred,
                    result,
                    (result * (1.0 - mask)) + target * mask,
                )
            ),
            "out/inpainting_result.png",
            nrow=4,
        )

    old_result = result
    old_target = target

    target = target - result.detach() + 0.5
    for inversion in tqdm.tqdm(
        inverter.all_inversion_steps(target * mask), total=len(inverter)
    ):
        result = inversion.ema.to_image()
        save_image(
            torch.cat(
                (
                    target,
                    mask,
                    target * mask,
                    inversion.final_pred,
                    result,
                    result + old_result,
                    (result + old_result * (1.0 - mask)) + old_target * mask,
                )
            ),
            "out/inpainting_result2.png",
            nrow=4,
        )
