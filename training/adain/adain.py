from .prelude import *

from .injection import *


__all__ = ["AdaIN", "AdaConv1x1", "AdaConv3x3"]


@dataclass(eq=False)
class AdaIN(nn.Module):
    """
    A reimplementation of Adaptive Instance Normalization.
    """

    style_size: int
    num_image_channels: int

    normalize: bool = True
    "If set false then the image won't be normalize before injecting the new style."

    inject_new_scale: bool = True
    "If set false then each channel will be left with a variance of zero."

    apply_skip_through_rescaling: bool = True
    "Conceptually the same thing as 'centering' random rescalings around the identity."

    inject_new_mean: bool = True
    "If set false then each channel will be left centered around zero."

    _normalize: nn.InstanceNorm2d = field(init=False)
    _injection: Injection = AdaINInjection()

    def __post_init__(self):
        super().__init__()
        self._normalize = nn.InstanceNorm2d(num_features=self.num_image_channels)

        # * Linear layers work just fine with ndim=3 tensors (it transforms the vectors of the last dim),
        # * just like if the first 2 dimensions were merged into a single one.
        self._mean_projection = nn.Linear(
            self.style_size,
            self.num_image_channels,
        )
        self._scale_projection = nn.Linear(
            self.style_size,
            self._injection.num_required_scale_features(self.num_image_channels),
        )

    def forward(self, image: ImageTensor, style: StyleTensor) -> ImageTensor:
        assert image.shape[1] == self.num_image_channels, image.shape
        assert style.shape[2] == self.style_size, style.shape

        # * Take as many style vectors as we need.
        # * (the style may have more because it is typically replicated between layers)

        assert style.shape[1] >= self.num_required_style_vectors()
        style = style[
            :, : self.num_required_style_vectors()
        ]  # todo random fixed permutation?

        if self.normalize:
            image = self._normalize(image)

        if self.inject_new_scale:
            scale = self._scale_projection(style)
            rescaled_image = self._injection.rescale(image, scale)

            if self.apply_skip_through_rescaling:
                rescaled_image = rescaled_image + image

            image = rescaled_image

        if self.inject_new_mean:
            mean = self._mean_projection(style[:, :1])
            image = self._injection.recenter(image, mean)

        return image

    def num_required_style_vectors(self) -> int:
        return self._injection.num_required_scale_vectors(self.num_image_channels)


@dataclass(eq=False)
class AdaConv1x1(AdaIN):
    """
    AdaIN with the rescaling replaced by a 1x1 convolution.
    """

    _injection: Injection = AdaConvInjection(kernel_size=1)  # @override

    @property
    def kernel_size(self) -> int:
        return self._injection.kernel_size


@dataclass(eq=False)
class AdaConv3x3(AdaConv1x1, AdaIN):
    """
    AdaIN with the rescaling replaced by a 3x3 convolution.
    """

    _injection: Injection = AdaConvInjection(kernel_size=3)  # @override


@dataclass(eq=False)
class AdaConv5x5(AdaConv1x1, AdaIN):
    """
    AdaIN with the rescaling replaced by a 5x5 convolution.
    """

    _injection: Injection = AdaConvInjection(kernel_size=5)  # @override
