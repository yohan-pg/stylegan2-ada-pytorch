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
# todo multigpu
# todo loss curves
# todo final vgg
# todo final fid
# todo save a validation batch

## todo ideas
# todo try encoding fakes instead? can we get that perfect?

## todo must haves
# todo measure ppl + fid on final interpolations (+ validatin data)
# todo improve persistence code to use their code snapshotting

## todo to test enventually
# todo train their W+
# todo review bn and wscale -> How to not do group norm?
# todo try training with a larger batch size

## todo nice to have
# todo loader workers
# todo LR schedule
# todo EMA

METHOD = "adaconv"
PKL_PATH = f"pretrained/tmp.pkl"

GAIN = 1.0
HEAD_GAIN = 1.0 if METHOD == "adaconv" else 1.0
VARIABLE_TYPE = WVariable
LEARNING_RATE = 1e-3 
FINE_TUNE_DISCRIMINATOR = True
BETA_1 = 0.0
BETA_2 = 0.999

BATCH_SIZE = 4
SUBSET_SIZE = None
DIST_WEIGHT = 1.0
DISCR_WEIGHT = 0.1

OUTDIR=f"out/encoder_{DISCR_WEIGHT}"

if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)

    G, D = open_generator_and_discriminator(PKL_PATH)

    encoder = Encoder(
        G,
        D,
        VARIABLE_TYPE,
        gain=GAIN,
        head_gain=HEAD_GAIN,
        lr=LEARNING_RATE,
        beta_1=BETA_1,
        beta_2=BETA_2,
        fine_tune_discriminator=FINE_TUNE_DISCRIMINATOR,
        discriminator_weight=DISCR_WEIGHT,
        distance_weight=DIST_WEIGHT
    )

    vgg = VGGCriterion()
    criterion = lambda preds, targets: vgg(preds, targets) 
    loader = EncodingDataset(path="./datasets/afhq256cat.zip").to_loader(
        batch_size=BATCH_SIZE,
        subset_size=SUBSET_SIZE,
    )

    for i, (preds, targets, loss) in enumerate(encoder.fit(loader, criterion)):
        if i % 10 == 0:
            print(loss.mean().item())
            save_image(
                encoder.make_prediction_grid(preds, targets),
                f"{OUTDIR}/encoding.png",
            )

        if i % 100 == 0:
            save_image(
                encoder.make_interpolation_grid(targets),
                f"{OUTDIR}/encoding_interpolation.png",
            )

        if i % 50_000 == 0:
            save_pickle(
                dict(E=encoder, G_ema=G, D=D),
                os.path.join(OUTDIR, f"encoder-snapshot-{i//1_000:06d}.pkl"),
            )
