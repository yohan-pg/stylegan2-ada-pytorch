from .prelude import *

from .injection import *

import numpy as np
from torch_utils import persistence
from torch_utils.ops import bias_act

__all__ = ["AdaIN", "AdaConv1x1", "AdaConv3x3", "FullyConnectedLayer"]


# ----------------------------------------------------------------------------


@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(
        self,
        in_features,  # Number of input features.
        out_features,  # Number of output features.
        bias=True,  # Apply additive bias before the activation function?
        activation="linear",  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier=1,  # Learning rate multiplier.
        bias_init=0,  # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(
            torch.randn([out_features, in_features]) / lr_multiplier
        )
        self.bias = (
            torch.nn.Parameter(torch.full([out_features], np.float32(bias_init)))
            if bias
            else None
        )
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == "linear" and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x


@dataclass(eq=False)
class AdaIN(nn.Module):
    """
    A reimplementation of Adaptive Instance Normalization.
    """
    style_size: int
    num_image_channels: int

    apply_skip_through_rescaling: bool = True
    
    _normalize: nn.InstanceNorm2d = field(init=False)
    _injection: Injection = AdaINInjection()

    def __post_init__(self):
        super().__init__()
        self._normalize = nn.InstanceNorm2d(num_features=self.num_image_channels)

        # * Linear layers work just fine with ndim=3 tensors (it transforms the vectors of the last dim),
        # * just like if the first 2 dimensions were merged into a single one.
        self._scale_projection = FullyConnectedLayer( 
            self.style_size,
            self._injection.num_required_scale_features(self.num_image_channels),
            bias=False
        )

        self.skip = nn.Parameter(torch.ones(self.num_image_channels, 1, 1))

    def forward(self, image: ImageTensor, style: StyleTensor) -> ImageTensor:
        assert image.shape[1] == self.num_image_channels, image.shape
        assert style.shape[2] == self.style_size, style.shape

        assert style.shape[1] >= self.num_required_style_vectors()
        style = style[
            :, : self.num_required_style_vectors()
        ]

        scale = self._scale_projection(style.reshape(-1, self.style_size)).reshape(style.shape)
        rescaled_image = self._injection.rescale(image, scale)

        if self.apply_skip_through_rescaling:
            return rescaled_image + image * self.skip
        else:
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
