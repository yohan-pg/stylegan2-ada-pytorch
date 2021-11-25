from edits import *

sys.path.append("DiffJPEG")
from inversion.radam import RAdam
from DiffJPEG import DiffJPEG

METHOD = "adaconv"
PATH = "out/unjpeg.png"
G_PATH = "pretrained/tmp.pkl"

BLUR = True

if __name__ == "__main__":
    G = open_generator(G_PATH).eval()
    E = open_encoder("out/encoder_0.1/encoder-snapshot-000050.pkl")
    target = open_target(
        G,
        "datasets/afhq2/test/cat/pixabay_cat_002905.png",
        "datasets/afhq2/test/cat/pixabay_cat_002997.png",
    )

    compress_jpeg = DiffJPEG(
        height=256, width=256, differentiable=True, quality=5
    ).cuda()  # todo pick out resolution from G
    compressed_target = compress_jpeg(target)

    edit(
        "unjpeg",
        G,
        E,
        target,
        lift=compress_jpeg,
        paste=lambda x: compressed_target - (compress_jpeg(x) - x),
        encoding_weight=0.07,
        truncation_weight=0.025,
    )
