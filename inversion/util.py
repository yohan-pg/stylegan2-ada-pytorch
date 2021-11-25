from .prelude import *

import kornia

def blur(x, iterations = 4):
    # return nn.
    for _ in range(iterations):
        x = kornia.filters.gaussian_blur2d(x, (3, 3), (1.0, 1.0)) 
    return x


def highpass(x, iterations = 8):
    return x - blur(x, iterations) + 0.5


def downsample(x, iters=2):
    for i in range(iters):
        # * Iteratively, aliasing isn't as bad apprantly
        x = F.interpolate(x, size=(x.shape[2] // 2, x.shape[3] // 2), mode="area")
    return x

def upsample(x, factor=4):
    return F.interpolate(x, size=(x.shape[2] * factor, x.shape[3] * factor), mode="bicubic", align_corners=False)

