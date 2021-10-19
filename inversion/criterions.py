from .prelude import *

import torchvision.transforms as TF
import functools


class InversionCriterion(nn.Module):
    pass  # todo this is just the interface of torch loss functions


class VGGCriterion(InversionCriterion):
    def __init__(self, on_crop: bool = False):
        super().__init__()

        self.on_crop = on_crop

        with dnnlib.util.open_url(
            "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt"
        ) as f:
            self.vgg16 = torch.jit.load(f).eval().cuda()

    # @functools.lru_cache(2) # todo
    def extract_features(self, x):
        "Expects an image with values between 0.0 and 1.0"
        if x.shape[2] > 256:
            x = F.interpolate(
                x, size=(256, 256), mode="area", align_corners=False
            )  #!! review this. especially mode="area"

        return self.vgg16(x.clone() * 255, resize_images=False, return_lpips=True)

    def forward(self, pred: ImageTensor, target: ImageTensor):
        if self.on_crop:
            pred_crop = TF.CenterCrop(128)(pred)
            target_crop = TF.CenterCrop(128)(target)
            return (
                (self.extract_features(pred) - self.extract_features(target)).square().sum().sum(dim=1)
            ) * 0.2 + (
                self.extract_features(pred_crop) - self.extract_features(target_crop)
            ).square().sum()
        return (
            (self.extract_features(pred) - self.extract_features(target)).square().sum(dim=1)
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