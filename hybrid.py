from edits import *

import kornia.filters

METHOD = "adaconv"
PATH = "out/hybrid.png"
G_PATH = "pretrained/tmp.pkl"

def lowpass(x, iterations = 100):
    for _ in range(iterations):
        x = kornia.filters.gaussian_lowpass2d(x, (3, 3), (1.0, 1.0)) 
    return x

if __name__ == "__main__":
    G = open_generator(G_PATH).eval()
    E = open_encoder("out/encoder_0.1/encoder-snapshot-000050.pkl")

    high_freq_target = open_target(
        G,
        "datasets/samples/high_freq_cat.png",
        "datasets/samples/high_freq_cat_2.png",
    )

    low_freq_target = open_target(
        G,
        "datasets/samples/low_freq_cat.png",
        "datasets/samples/low_freq_cat_2.png",
    )

    high_freqs = high_freq_target - lowpass(high_freq_target)
    low_freqs = lowpass(low_freq_target)
    target = high_freqs + low_freqs

    edit(
        "hybrid",
        G,
        E,
        target,
        paste=lambda x: lowpass(x) + high_freqs,
        encoding_weight=0.2,
        truncation_weight=0.05,
    )