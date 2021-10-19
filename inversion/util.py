from .prelude import *

def downsample(x, factor=4):
    return F.interpolate(x, size=(x.shape[2] // factor, x.shape[3] // factor), mode="bicubic", align_corners=False)

def upsample(x, factor=4):
    return F.interpolate(x, size=(x.shape[2] * factor, x.shape[3] * factor), mode="bicubic", align_corners=False)

