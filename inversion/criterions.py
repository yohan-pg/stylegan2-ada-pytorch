from .prelude import *

import functools


class InversionCriterion(nn.Module):
    pass # todo this is just the interface of torhc loss functions


class VGGCriterion(InversionCriterion):
    def __init__(self):
        super().__init__()

        with dnnlib.util.open_url(
            "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt"
        ) as f:
            self.vgg16 = torch.jit.load(f).eval().cuda()

    # @functools.lru_cache(2) # todo
    def extract_features(self, x):
        "Expects an image with values between 0.0 and 1.0"
        if x.shape[2] > 256:
            x = F.interpolate(x, size=(256, 256), mode="area", align_corners=False)

        return self.vgg16(
            x.clone() * 255, resize_images=False, return_lpips=True
        )

    def forward(self, pred: ImageTensor, target: ImageTensor):
        return (
            (self.extract_features(pred) - self.extract_features(target)).square().sum()
        )  #!? sum not mean?
