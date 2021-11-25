# Taken from https://github.com/genforce/idinvert_pytorch/blob/master/models/stylegan_encoder_network.py

"""Contains the implementation of encoder for StyleGAN inversion.
For more details, please check the paper:
https://arxiv.org/pdf/2004.00049.pdf
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

__all__ = ["StyleGANEncoderNet"]

# Resolutions allowed.
_RESOLUTIONS_ALLOWED = [8, 16, 32, 64, 128, 256, 512, 1024]

# Initial resolution.
_INIT_RES = 4


class StyleGANEncoderNet(nn.Module):
    """Defines the encoder network for StyleGAN inversion.
    NOTE: The encoder takes images with `RGB` color channels and range [-1, 1]
    as inputs, and encode the input images to W+ space of StyleGAN.
    """
    
    @staticmethod
    def configure_for(G, **kwargs):
        return StyleGANEncoderNet(
            w_space_dim=G.w_dim,
            resolution=G.img_resolution, 
            num_style_vectors=G.num_required_vectors(),
            **kwargs
        )

    def __init__(
        self,
        resolution,
        w_space_dim,
        num_style_vectors,
        image_channels=3,
        encoder_channels_base=64,
        encoder_channels_max=1024,
        use_wscale=True,
        head_gain: float = 1.0,
        w_plus=True,
        use_bn=True,
        predict_residual_from_mean=True
    ):
        """Initializes the encoder with basic settings.
        Args:
          resolution: The resolution of the input image.
          w_space_dim: The dimension of the disentangled latent vectors, w.
            (default: 512)
          image_channels: Number of channels of the input image. (default: 3)
          encoder_channels_base: Base factor of the number of channels used in
            residual blocks of encoder. (default: 64)
          encoder_channels_max: Maximum number of channels used in residual blocks
            of encoder. (default: 1024)
          use_wscale: Whether to use `wscale` layer. (default: False)
          use_bn: Whether to use batch normalization layer. (default: False)
        Raises:
          ValueError: If the input `resolution` is not supported.
        """
        super().__init__()

        if resolution not in _RESOLUTIONS_ALLOWED:
            raise ValueError(
                f"Invalid resolution: {resolution}!\n"
                f"Resolutions allowed: {_RESOLUTIONS_ALLOWED}."
            )

        self.init_res = _INIT_RES
        self.resolution = resolution
        self.w_space_dim = w_space_dim
        self.image_channels = image_channels
        self.encoder_channels_base = encoder_channels_base
        self.encoder_channels_max = encoder_channels_max
        self.use_wscale = use_wscale
        self.use_bn = use_bn
        self.head_gain = head_gain
        self.num_style_vectors = num_style_vectors
        # Blocks used in encoder.
        self.num_blocks = int(np.log2(resolution))
        # Layers used in generator.
        self.num_layers = int(np.log2(self.resolution // self.init_res * 2)) * 2

        in_channels = self.image_channels
        out_channels = self.encoder_channels_base
        for block_idx in range(self.num_blocks):
            if block_idx == 0:
                self.add_module(
                    f"block{block_idx}",
                    FirstBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        use_wscale=self.use_wscale,
                        use_bn=self.use_bn,
                    ),
                )

            elif block_idx == self.num_blocks - 1:
                in_channels = in_channels * self.init_res * self.init_res
                assert not (w_plus and self.num_style_vectors > 1)
                
                out_channels = self.w_space_dim 
                if w_plus:
                    out_channels *= 2 * block_idx
                # if self.num_style_vectors > 1:
                #     out_channels *= 4

                self.add_module(
                    f"block{block_idx}",
                    LastBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        use_wscale=True,
                        use_bn=self.use_bn,
                    ),
                )

                if num_style_vectors > 1:
                    self.heads = nn.Sequential(
                        # Splitter(num_style_vectors, w_space_dim, out_channels, head_gain),
                        # nn.ReLU(),
                        Splitter(num_style_vectors, w_space_dim, out_channels, head_gain),
                    )
                    # self.heads = nn.TransformerEncoder(nn.TransformerEncoderLayer(512, 8, dropout = 0), 4)
                    # self.embeddings = nn.Parameter(torch.randn(1, 512, 512).cuda() / math.sqrt(self.w_space_dim))

            else:
                self.add_module(
                    f"block{block_idx}",
                    ResBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        use_wscale=self.use_wscale,
                        use_bn=self.use_bn,
                    ),
                )
            in_channels = out_channels
            out_channels = min(out_channels * 2, self.encoder_channels_max)

        self.downsample = AveragePoolingLayer()
        self.predict_residual_from_mean = predict_residual_from_mean

        if self.predict_residual_from_mean:
            self.init_mean = None
        else:
            self.init_mean = torch.randn(1, 512, 512).cuda()

    def _forward(self, x):
        x = x * 2 - 1 

        for block_idx in range(self.num_blocks):
            if 0 < block_idx < self.num_blocks - 1:
                x = self.downsample(x)
            x = self.__getattr__(f"block{block_idx}")(x)
            
        if self.num_style_vectors > 1:
            return self.heads(x)
            # return self.heads((x.reshape(-1, 1, 512).repeat(1, 512, 1) + self.embeddings).transpose(0, 1)).transpose(0, 1)
        else:
            return x.reshape(x.shape[0], -1, self.w_space_dim)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (
            self.image_channels,
            self.resolution,
            self.resolution,
        ):
            raise ValueError(
                f"The input image should be with shape [batch_size, "
                f"channel, height, width], where "
                f"`channel` equals to {self.image_channels}, "
                f"`height` and `width` equal to {self.resolution}!\n"
                f"But {x.shape} is received!"
            )
        
        with torch.no_grad():
            if self.init_mean is None and self.predict_residual_from_mean:
                self.init_mean = self._forward(x).mean(dim=0, keepdim=True)

        return self._forward(x) - (0.0 if self.init_mean is None else self.init_mean)


class Splitter(nn.Module):
    def __init__(self, num_style_vectors, w_space_dim, out_channels, head_gain):
        super().__init__()
        self.heads = nn.Parameter(
            torch.randn(
                num_style_vectors,
                w_space_dim,
                out_channels, 
            ) / head_gain
        ) 
        self.head_gain = head_gain

        with torch.no_grad():
            # * kaiming normal init on with fan in & fan out
            self.heads /= math.sqrt((out_channels + w_space_dim) / 2)

    def forward(self, x):
        return (self.heads @ x.t()).transpose(2, 1).transpose(1, 0) * self.head_gain

class AveragePoolingLayer(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        ksize = [self.scale_factor, self.scale_factor]
        strides = [self.scale_factor, self.scale_factor]
        return F.avg_pool2d(x, kernel_size=ksize, stride=strides, padding=0)


class BatchNorm(nn.Module):
    def __init__(self, channels, gamma=False, beta=True, decay=0.9, epsilon=1e-5):
        """Initializes with basic settings.
        Args:
          channels: Number of channels of the input tensor.
          gamma: Whether the scale (weight) of the affine mapping is learnable.
          beta: Whether the center (bias) of the affine mapping is learnable.
          decay: Decay factor for moving average operations in this layer.
          epsilon: A value added to the denominator for numerical stability.
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(
            num_features=channels,
            affine=True,
            track_running_stats=True,
            momentum=1 - decay,
            eps=epsilon,
        )
        self.bn.weight.requires_grad = gamma
        self.bn.bias.requires_grad = beta

    def forward(self, x):
        return self.bn(x)


class WScaleLayer(nn.Module):
    """Implements the layer to scale weight variable and add bias.
    NOTE: The weight variable is trained in `nn.Conv2d` layer (or `nn.Linear`
    layer), and only scaled with a constant number, which is not trainable in
    this layer. However, the bias variable is trainable in this layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size, gain=np.sqrt(2.0)):
        super().__init__()
        fan_in = in_channels * kernel_size * kernel_size
        self.scale = gain / np.sqrt(fan_in)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        if x.ndim == 4:
            return x * self.scale + self.bias.view(1, -1, 1, 1)
        if x.ndim == 2:
            return x * self.scale + self.bias.view(1, -1)
        raise ValueError(
            f"The input tensor should be with shape [batch_size, "
            f"channel, height, width], or [batch_size, channel]!\n"
            f"But {x.shape} is received!"
        )


class FirstBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_wscale=False,
        wscale_gain=np.sqrt(2.0),
        use_bn=False,
        activation_type="lrelu",
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.scale = wscale_gain / np.sqrt(in_channels * 3 * 3) if use_wscale else 1.0
        self.bn = nn.GroupNorm(1, out_channels) if use_bn else nn.Identity()

        if activation_type == "linear":
            self.activate = nn.Identity()
        elif activation_type == "lrelu":
            self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            raise NotImplementedError(
                f"Not implemented activation function: " f"{activation_type}!"
            )

    def forward(self, x):
        return self.activate(self.bn(self.conv(x) * self.scale))


class ResBlock(nn.Module):
    """
    Usually, each residual block contains two convolutional layers, each of which
    is followed by batch normalization layer and activation layer.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        use_wscale=False,
        wscale_gain=np.sqrt(2.0),
        use_bn=False,
        activation_type="lrelu",
    ):
        """Initializes the class with block settings.
        Args:
          in_channels: Number of channels of the input tensor fed into this block.
          out_channels: Number of channels of the output tensor.
          kernel_size: Size of the convolutional kernels.
          stride: Stride parameter for convolution operation.
          padding: Padding parameter for convolution operation.
          use_wscale: Whether to use `wscale` layer.
          wscale_gain: The gain factor for `wscale` layer.
          use_bn: Whether to use batch normalization layer.
          activation_type: Type of activation. Support `linear` and `lrelu`.
        Raises:
          NotImplementedError: If the input `activation_type` is not supported.
        """
        super().__init__()

        # Add shortcut if needed.
        if in_channels != out_channels:
            self.add_shortcut = True
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.scale = wscale_gain / np.sqrt(in_channels) if use_wscale else 1.0
            self.bn = nn.GroupNorm(1, out_channels) if use_bn else nn.Identity()
        else:
            self.add_shortcut = False
            self.identity = nn.Identity()

        hidden_channels = min(in_channels, out_channels)

        # First convolutional block.
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.scale1 = 1.0 if use_wscale else wscale_gain / np.sqrt(in_channels * 3 * 3)
        # NOTE: WScaleLayer is employed to add bias.
        self.wscale1 = WScaleLayer(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            gain=wscale_gain,
        )
        self.bn1 = nn.GroupNorm(1, hidden_channels) if use_bn else nn.Identity()

        # Second convolutional block.
        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.scale2 = (
            1.0 if use_wscale else wscale_gain / np.sqrt(hidden_channels * 3 * 3)
        )
        self.wscale2 = WScaleLayer(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=3,
            gain=wscale_gain,
        )
        self.bn2 = nn.GroupNorm(1, out_channels) if use_bn else nn.Identity()

        if activation_type == "linear":
            self.activate = nn.Identity()
        elif activation_type == "lrelu":
            self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            raise NotImplementedError(
                f"Not implemented activation function: " f"{activation_type}!"
            )

    def forward(self, x):
        if self.add_shortcut:
            y = self.activate(self.bn(self.conv(x) * self.scale))
        else:
            y = self.identity(x)
        x = self.activate(self.bn1(self.wscale1(self.conv1(x) / self.scale1)))
        x = self.activate(self.bn2(self.wscale2(self.conv2(x) / self.scale2)))
        return x + y


class LastBlock(nn.Module):
    """Implements the last block, which is a dense block."""

    def __init__(
        self, in_channels, out_channels, use_wscale=False, wscale_gain=1.0, use_bn=False
    ):
        super().__init__()

        self.fc = nn.Linear(
            in_features=in_channels, out_features=out_channels, bias=False
        )
        self.scale = wscale_gain / np.sqrt(in_channels) if use_wscale else 1.0
        self.bn = nn.GroupNorm(1, out_channels) if use_bn else nn.Identity()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc(x) * self.scale
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        return self.bn(x).view(x.shape[0], x.shape[1])
