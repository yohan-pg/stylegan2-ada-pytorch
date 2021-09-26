# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from inversion import *
from encoding import *
import torchvision

## todo priority
# todo try on 256 to see if blurriness is better
# todo first get a training going
# todo try with an untrained generator
# todo try encoding fakes instead? can we get that perfect?

## todo core
# todo integrate with the generator training
# todo first fine-tune the generator while training the encoder (fixed mapper) -> were the params really updated?
# todo measure recall after that, make sure it improved
# todo try optimizaion after that? is it now fixed?
# todo then integrate the encoder inside the generator training loop
# todo can we first try something similar inside our training loop?
# todo after that, consider adding discriminator in style space, so the distribution matches?
# todo try adding in the discriminator fine-tuning
# todo implement the in-domain encoder optimization regularization

## todo must haves
# todo save the loss graph
# todo add metrics
# todo a reconstruction metric (vgg)
# todo interpolation metrics (?? ppl + ?)
# todo use validation data
# todo build the dataset
# todo images
# todo metrics
# todo improve persistence code to use their code snapshotting
# todo better saved folders (just like training runs)

## todo to test enventually
# todo review bn and wscale
# todo try training with a larger batch size
# todo study the impact of gain and # of flattened features on the output quality
# todo try our method on W+, their method on W, etc.
# todo WTH, we can interpolate after overfitting a single batch!?

## todo nice to have
# todo loader workers
# todo multigpu
# todo LR schedule
# todo EMA

#! number of features for adaconv has yet to be optimized

METHOD = "adain"

PKL_PATH = f"pretrained/alpha-{METHOD}-002600.pkl"
GAIN = 1 / math.sqrt(512) if METHOD == "adaconv" else 1.0 
HEAD_GAIN = 1.0 if METHOD == "adaconv" else 1.0
VARIABLE_TYPE = WVariable if METHOD == "adaconv" else WPlusVariable
LEARNING_RATE = 1e-3 
FINE_TUNE_GENERATOR = False
FINE_TUNE_DISCRIMINATOR = False
BETA_1 = 0.0
BETA_2 = 0.0

BATCH_SIZE = 4
SUBSET_SIZE = None
DISCR_WEIGHT = 0.0


if __name__ == "__main__":
    G, D = open_generator(PKL_PATH), None

    encoder = Encoder(
        G,
        D,
        VARIABLE_TYPE,
        gain=GAIN,
        head_gain=HEAD_GAIN,
        lr=LEARNING_RATE,
        beta_1=BETA_1,
        beta_2=BETA_2,
        fine_tune_generator=FINE_TUNE_GENERATOR,
        fine_tune_discriminator=FINE_TUNE_DISCRIMINATOR,
    )

    vgg = VGGCriterion()
    # fake_targets = torch.cat(
    #     (
    #         open_target(G, "./datasets/samples/cats/00000/img00000009.png"),
    #         open_target(G, "./datasets/samples/cats/00000/img00000010.png"),
    #     )
    # )
    criterion = lambda preds, targets: vgg(preds, targets) 
    # + DISCR_WEIGHT * torch.nn.functional.softplus(
    # -D(preds, None)
    # ).mean()
    loader = EncodingDataset(path="./datasets/afhq128cat.zip").to_loader(
        batch_size=BATCH_SIZE,
        subset_size=SUBSET_SIZE,
    )

    for i, (preds, targets, loss) in enumerate(encoder.fit(loader, criterion)):
        if i % 10 == 0:
            print(loss.item())
            save_image(
                encoder.make_prediction_grid(preds, targets),
                "out/encoding.png",
            )

        if i % 100 == 0:
            save_image(
                encoder.make_interpolation_grid(targets),
                f"out/encoding_interpolation.png",
            )

        if i % 50_000 == 0:
            save_pickle(
                dict(E=encoder, G_ema=G, D=D),  # todo ema?
                os.path.join("out", f"encoder-snapshot-{i//1_000:06d}.pkl"),
            )
