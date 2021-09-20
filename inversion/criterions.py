from .prelude import *


class InversionCriterion(nn.Module):
    pass


class VGGCriterion(InversionCriterion):
    def __init__(self, target):
        # Load VGG16 feature detector.
        url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt"
        with dnnlib.util.open_url(url) as f:
            self.vgg16 = torch.jit.load(f).eval().cuda()

        # Features for target image.
        target = target.unsqueeze(0).cuda().to(torch.float32)
        if target.shape[2] > 256:
            target = F.interpolate(target, size=(256, 256), mode="area")

        self.target_features = self.vgg16(
            target.clone(), resize_images=False, return_lpips=True
        )

    def forward(self, synth_images):
        synth_images = (synth_images + 1) * (255 / 2)

        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode="area")

        synth_features = self.vgg16(
            synth_images, resize_images=False, return_lpips=True
        )
        return (self.target_features - synth_features).square().sum()  #!? sum not mean?


class L1Criterion(InversionCriterion):
    def __call__(self, synth_images, target_images):
        return torch.nn.L1Loss()((synth_images + 1) / 2, target_images / 255.0)
