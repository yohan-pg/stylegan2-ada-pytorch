from .prelude import *


class InversionCriterion(nn.Module):
    def __rmul__(self, weight):
        return WeightedCriterion(self, weight)

    def __add__(self, other):
        return CombinedCriterion(self, other)

    def transform(self, f):
        return TransformedCriterion(self, f)

    def transform_target(self, f):
        return TargetTransformedCriterion(self, f)


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
        #     )  #!!! review this. especially mode="area"

        return self.vgg16(x.clone() * 255, resize_images=False, return_lpips=True)

    def forward(self, pred: ImageTensor, target: ImageTensor):
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)

        loss = (pred_features - target_features).square().sum(dim=1)
        #!? sum not mean?
        return loss


class NullCriterion(InversionCriterion):
    def forward(self, pred: ImageTensor, _: ImageTensor):
        return (pred * 0).sum(dim=(1, 2, 3))


class DistanceCriterion(InversionCriterion):
    def __init__(self, distance):
        super().__init__()
        self.distance = distance

    def forward(self, pred: ImageTensor, target: ImageTensor):
        return self.distance(pred, target)


class TransformedCriterion(InversionCriterion):
    def __init__(self, criterion, f):
        super().__init__()
        self.criterion = criterion
        self.f = f

    def forward(self, pred: ImageTensor, target: ImageTensor):
        return self.criterion(self.f(pred), target)


class TargetTransformedCriterion(InversionCriterion):
    def __init__(self, criterion, f):
        super().__init__()
        self.criterion = criterion
        self.f = f

    def forward(self, pred: ImageTensor, target: ImageTensor):
        return self.criterion(pred, self.f(pred, target))


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
