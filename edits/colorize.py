from inversion import *
from edits import *

from kornia.color import RgbToHsv

METHOD = "adaconv"
PATH = "out/colorize.png"
G_PATH = "pretrained/tmp.pkl"

if __name__ == "__main__":
    G = open_generator(G_PATH).eval()
    E = open_encoder("out/encoder_0.1/encoder-snapshot-000050.pkl")

    target = open_target(
        G,
        "datasets/afhq2/test/cat/pixabay_cat_002905.png",
        "datasets/afhq2_cat256_test/00000/img00000162.png",
        "datasets/afhq2_cat256_test/00000/img00000195.png",
    )

    def to_grayscale(x):
        return x.mean(dim=1, keepdim=True).repeat_interleave(3, dim=1)

    grayscale_target = to_grayscale(target)

    def paste(x):
        return grayscale_target + x - to_grayscale(x)

    def image_saturation(x):
        return x.std(dim=1, keepdim=True)

    def saturation_loss(pred):
        return -image_saturation(pred).mean()

    edit(
        "colorize",
        G,
        E,
        target,
        lift=to_grayscale,
        paste=paste,
        encoding_weight=0.2,
        truncation_weight=0.05,
        penalty = saturation_loss
    )
