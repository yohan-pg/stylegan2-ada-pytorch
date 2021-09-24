
def downsample(x):
    return F.interpolate(x, size=(x.shape[2] // 2, x.shape[3] // 2), mode="bicubic", align_corners=False)

def upsample(x):
    return F.interpolate(x, size=(x.shape[2] * 2, x.shape[3] * 2), mode="bicubic", align_corners=False)

