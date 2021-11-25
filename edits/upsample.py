from inversion import *
from edits import *

METHOD = "adaconv"
PATH = "out/upsample.png"
G_PATH = "pretrained/tmp.pkl"

BLUR = True
SCALE = 40 

if __name__ == "__main__":
    G = open_generator(G_PATH).eval()
    D = open_discriminator(G_PATH)
    E = open_encoder("out/encoder_0.1/encoder-snapshot-000050.pkl")

    target = open_target(
        G,
        "datasets/afhq2/test/cat/flickr_cat_000176.png",
    )

    def lift(x):
        return blur(x, SCALE)

    def present(x):
        return F.interpolate(x, scale_factor=SCALE)

    def paste(x):
        return x - lift(x) + lift(target)

    edit(
        "upsample",
        G,
        E,
        target, 
        lift=lift,
        paste=paste,
        present=present, 
        truncation_weight=0.05, 
        encoding_weight=0.35,
    )

