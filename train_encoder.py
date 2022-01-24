from inversion import *
from encoding import *
from datetime import datetime

## todo must haves
# todo try IID machines
# todo try a chunking version of splitter
# todo chunking splitter is too slow -> vectorize

## todo to test eventually
# todo data augmentation? we are overfitting. try ffhq first tho -> at least x flips
# todo try training with a larger batch size. Is the quality better?
# todo find a better solution for w+? why is it so bad?
# todo try increasing the final style size. going from a single 512 vector to 512 other vectors of the same size is a bit crazy
# todo try a better architecture (residual?)
# todo try encoding fakes instead? can we get that perfect?
# todo review ID invert training parameters
# todo hyperparam search for the ideal discr_weight


NAME = "encoder_sanity_check"
METHOD, PKL_PATH = "adaconv", f"pretrained/no_torgb_adaconv_tmp.pkl"

GAIN = 1.0 
VARIABLE_TYPE = WVariable
LEARNING_RATE = 1e-3

BATCH_SIZE = 4
SUBSET_SIZE = None
DIST_WEIGHT = 1.0
DISCR_WEIGHT = 0.1
MEAN_REG = 0.0 

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
        lr=LEARNING_RATE,
        discriminator_weight=DISCR_WEIGHT,
        distance_weight=DIST_WEIGHT,
        mean_regularization_weight=MEAN_REG,
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

    shutil.copyfile(__file__, OUTDIR + "/launch_config.py")
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
