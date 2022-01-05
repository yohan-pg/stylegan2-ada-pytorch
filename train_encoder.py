from inversion import *
from encoding import *
from datetime import datetime

## todo must haves
# todo why does the single layer fail again?
# todo vectorize w+
# todo double check if interpolation is good now
# todo if it isnt, try to add a w_mean regularization, in order to make interpolation work again? 

## todo to test eventually
# todo find a better solution for w+? why is it so bad?
# todo try training with a larger batch size. Is the quality better?
# todo try increasing the final style size. going from a single 512 vector to 512 other vectors of the same size is a bit crazy
# todo try encoding fakes instead? can we get that perfect?
# todo try RAdam to avoid going crazy early in training?
# todo try predicting S variable?
# todo hyperparam search for the ideal discr_weight
# todo try a better architecture (residual?)

NAME = "encoder_0.0"
METHOD, PKL_PATH = "adaconv", f"pretrained/no_torgb_adaconv_tmp.pkl"

GAIN = 1.0
HEAD_GAIN = 1.0 if METHOD == "adaconv" else 1.0
VARIABLE_TYPE = WVariable
LEARNING_RATE = 1e-3 
BETA_1 = 0.0
BETA_2 = 0.999

BATCH_SIZE = 4
SUBSET_SIZE = None 
DIST_WEIGHT = 1.0
DISCR_WEIGHT = 0.0 

OUTDIR = f"encoder-training-runs/{NAME}/" + str(datetime.now()).split(".")[0].replace(
    " ", "_"
)

if __name__ == "__main__":
    os.makedirs(
        OUTDIR,
    )

    G, D, training_set_kwargs = open_generator_and_discriminator(PKL_PATH)

    encoder = Encoder(
        G,
        D,
        VARIABLE_TYPE,
        gain=GAIN,
        head_gain=HEAD_GAIN,
        lr=LEARNING_RATE,
        beta_1=BETA_1,
        beta_2=BETA_2,
        discriminator_weight=DISCR_WEIGHT,
        distance_weight=DIST_WEIGHT,
        single_layer_adaconv=False
    )

    vgg = VGGCriterion()
    criterion = lambda preds, targets: vgg(preds, targets)
    loader = EncodingDataset(path=training_set_kwargs["path"]).to_loader(
        batch_size=BATCH_SIZE,
        subset_size=SUBSET_SIZE,
    )
    validation_loader = EncodingDataset(
        path=training_set_kwargs["path"].replace(".zip", "_test.zip")
    ).to_loader(
        batch_size=BATCH_SIZE,
        subset_size=None,
    )
    validation_targets = next(iter(validation_loader)).cuda()
    writer = launch_tensorboard(OUTDIR)

    print("Starting training...")
    for i, (preds, targets, loss) in enumerate(
        tqdm.tqdm(encoder.fit(loader, criterion))
    ):
        def save_images(suffix):
            save_image(
                encoder.make_prediction_grid(preds, targets),
                f"{OUTDIR}/encoding{suffix}.png",
            )
            if BATCH_SIZE > 1:
                save_image(
                    encoder.make_interpolation_grid(targets),
                    f"{OUTDIR}/encoding_interpolation{suffix}.png",
                )
            save_image(
                encoder.make_prediction_grid(
                    encoder(validation_targets).to_image(), validation_targets
                ),
                f"{OUTDIR}/encoding_validation{suffix}.png",
            )
            if BATCH_SIZE > 1:
                save_image(
                    encoder.make_interpolation_grid(validation_targets),
                    f"{OUTDIR}/encoding_interpolation_validation{suffix}.png",
                )

        with torch.no_grad():
            writer.add_scalar("Loss/train", loss.mean().item(), i)

            if i % 100 == 0:
                writer.add_scalar(
                    "Loss/validation",
                    encoder.evaluate(targets, criterion)[-1].mean().item(),
                    i,
                )
                save_images("")

            if i % 50_000 == 0:
                tag = f"{i//1_000:06d}"
                save_images(tag)
                save_pickle(
                    dict(E=encoder, G_ema=G, D=D),
                    os.path.join(OUTDIR, f"encoder-snapshot-{tag}.pkl"),
                )
