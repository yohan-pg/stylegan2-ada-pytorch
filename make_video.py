from inversion import *

from training.dataset import ImageFolderDataset
import training.networks as networks
import itertools 

OUTDIR = "out/video"


if __name__ == "__main__":
    fresh_dir(OUTDIR)

    adaconv = open_generator("pretrained/adaconv-256-001200.pkl")
    adain = open_generator("pretrained/adain-256-001200.pkl")
    target = open_target(adaconv, "datasets/afhq2/test/cat/pixabay_cat_002865.png")
    
    inversion_adaconv = Inverter(
        adaconv,
        num_steps=500,
        learning_rate=0.01,
        variable_type=WPlusVariable,
        snapshot_frequency=1
    )(target)

    inversion_adain = Inverter(
        adain,
        num_steps=500,
        learning_rate=0.01,
        variable_type=WPlusVariable,
        snapshot_frequency=1
    )(target)

    Inversion.save_to_video("out/video.mp4", [
        inversion_adaconv,
        inversion_adain
    ])
    