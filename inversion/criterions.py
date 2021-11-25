from .prelude import *

import torchvision.transforms as TF
import functools


class InversionCriterion(nn.Module):
    def __rmul__(self, weight):
        return WeightedCriterion(self, weight)

    def __add__(self, other):
        return CombinedCriterion(self, other)

    def transform(self, f):
        return TransformedCriterion(self, f)


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
        # if x.shape[2] > 256:
        #     x = F.interpolate(
        #         x, size=(256, 256), mode="area", align_corners=False
        #     )  #!! review this. especially mode="area"

        return self.vgg16(x.clone() * 255, resize_images=False, return_lpips=True)

    def forward(self, pred: ImageTensor, target: ImageTensor):
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)

        loss = (pred_features - target_features).square().sum(dim=1)
        #!? sum not mean?
        return loss


class HighLevelVGGCriterion(InversionCriterion):
    vgg = None

    def __init__(self):
        super().__init__()

        if HighLevelVGGCriterion.vgg is None:
            HighLevelVGGCriterion.vgg = torch.hub.load("pytorch/vision:v0.10.0", "vgg19", pretrained=True).eval().cuda().features[0:3]
            # [0:6]
            # (
            #     torch.hub.load(
            #         "pytorch/vision:v0.10.0", "inception_v3", pretrained=True
            #     )
            #     .eval()
            #     .cuda()
            # )
        self.normalize = TF.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def extract_features(self, x):
        "Expects an image with values between 0.0 and 1.0"
        features = self.vgg(self.normalize(x.clone()))
        return features

    def forward(self, pred: ImageTensor, target: ImageTensor):
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)

        loss = (pred_features - target_features).abs().sum(dim=1)
        #!? sum not mean?
        return loss


class DiscriminatorCriterion(InversionCriterion):
    def __init__(self, D):
        super().__init__()
        self.D = D

    def forward(self, pred: ImageTensor, _: ImageTensor):
        return self.D(pred, None)



class NullCriterion(InversionCriterion):
    def forward(self, pred: ImageTensor, _: ImageTensor):
        return (pred * 0).sum(dim=(1, 2, 3))



class MultiscaleVGGCriterion(VGGCriterion):
    def __init__(self, num_downscales=4):
        self.num_downscales = num_downscales

    def forward(self, pred: ImageTensor, target: ImageTensor):
        loss = 0.0

        for i in range(self.num_downscales):

            def downsample(x):
                return F.interpolate(
                    x,
                    scale_factor=1 / 2 ** i,
                    mode="bilinear",
                    recompute_scale_factor=False,
                )

            pred_features = self.extract_features(downsample(pred))
            target_features = self.extract_features(downsample(target))
            loss += ((pred_features - target_features).square().sum(dim=1)) * 2 ** i
        #!? sum not mean?
        return loss / self.num_downscales


class TransformedCriterion(InversionCriterion):
    def __init__(self, criterion, f):
        super().__init__()
        self.criterion = criterion
        self.f = f

    def forward(self, pred: ImageTensor, target: ImageTensor):
        return self.criterion(self.f(pred), self.f(target))


class HalfTransformedCriterion(InversionCriterion):
    def __init__(self, criterion, f):
        super().__init__()
        self.criterion = criterion
        self.f = f

    def forward(self, pred: ImageTensor, target: ImageTensor):
        return self.criterion(pred, self.f(target))


class CombinedCriterion(InversionCriterion):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, pred: ImageTensor, target: ImageTensor):
        return self.a(pred, target) + self.b(pred, target)


class WeightedCriterion(InversionCriterion):
    def __init__(self, criterion, weight):
        super().__init__()
        self.criterion = criterion
        self.weight = weight

    def forward(self, pred: ImageTensor, target: ImageTensor):
        return self.criterion.forward(pred, target) * self.weight
