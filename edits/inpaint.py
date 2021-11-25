from inversion import *

from edits import *

from spectral import *
from training.networks import Discriminator

import torch.optim.lr_scheduler as lr_scheduler

from skimage import data
from skimage import exposure


def select_right_half(x):
    return x[:, :, :, x.shape[-1] // 2 :]


def pad_left_half(x):
    return torch.cat((torch.zeros_like(x), x), dim=3)

def draw_mask(x):
    x = x.clone()
    mask = torch.zeros_like(x)
    region = select_right_half(mask)
    region += 1
    return mask

if __name__ == "__main__":
    G = open_generator("pretrained/tmp.pkl")
    E = open_encoder("out/encoder_0.1/encoder-snapshot-000050.pkl")
    target = open_target(
        G,
        "datasets/afhq2/test/cat/flickr_cat_000176.png",
        "datasets/afhq2/test/cat/flickr_cat_000236.png",
        "datasets/afhq2/test/cat/pixabay_cat_000117.png",
    )

    def paste(x):
        result = x.clone()
        select_right_half(result).copy_(select_right_half(target))
        return result

    edit(
        "inpaint",
        G, 
        E, target, 
        lift=select_right_half,
        present=pad_left_half,
        paste=paste,
        encoding_weight=0.2,
        truncation_weight=0.05,
    )