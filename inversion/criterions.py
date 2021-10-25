from .prelude import *

import torchvision.transforms as TF
import functools


class InversionCriterion(nn.Module):
    pass  # todo this is just the interface of torch loss functions


class VGGCriterion(InversionCriterion):
    vgg16 = None

    def __init__(self):
        super().__init__()

        if VGGCriterion.vgg16 is None:
            with dnnlib.util.open_url(
                "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt"
            ) as f:
                VGGCriterion.vgg16 = torch.jit.load(f).eval().cuda()

    # @functools.lru_cache(2) # todo
    def extract_features(self, x):
        "Expects an image with values between 0.0 and 1.0"
        if x.shape[2] > 256:
            x = F.interpolate(
                x, size=(256, 256), mode="area", align_corners=False
            )  #!! review this. especially mode="area"

        return self.vgg16(x.clone() * 255, resize_images=False, return_lpips=True)

    def forward(self, pred: ImageTensor, target: ImageTensor):
        return (
            (self.extract_features(pred) - self.extract_features(target))
            .square()
            .sum(dim=1)
        )
        #!? sum not mean?


class MaskedVGGCriterion(VGGCriterion):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def forward(self, pred: ImageTensor, target: ImageTensor):
        return super().forward(pred * self.mask, target * self.mask)


class DownsamplingVGGCriterion(VGGCriterion):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, pred: ImageTensor, target: ImageTensor):
        return super().forward(self.f(pred), self.f(target))


class VGGCriterionWithNoise(VGGCriterion):
    def __init__(self, amount=0.01):
        super().__init__()
        self.amount = amount

    def forward(self, pred: ImageTensor, target: ImageTensor):
        return super().forward(
            pred + torch.randn_like(pred) * self.amount,
            target + torch.rand_like(target) * self.amount,
        )
