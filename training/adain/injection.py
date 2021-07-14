from .prelude import *


class Injection:
    def num_required_scale_vectors(self, num_image_channels: int) -> int:
        "Given this # of channels, how many scale vectors should we pull out of the mapper per image?"
        raise NotImplementedError

    def num_required_scale_features(self, num_image_channels: int) -> int:
        "Given this # of channels, how many features should a scale vector (after projection) have?"
        raise NotImplementedError

    def _validate_scale_shape(self, image: ImageTensor, scale: ScaleTensor):
        "Verifies that the scale tensor has the correct shape for this image (the typing is insufficient)."
        assert scale.shape[1] == self.num_required_scale_vectors(image.shape[1])
        assert scale.shape[2] == self.num_required_scale_features(image.shape[1])

    def rescale(self, image: ImageTensor, scale: ScaleTensor) -> ImageTensor:
        "Injects the new scale into the image."
        raise NotImplementedError

    def recenter(self, image: ImageTensor, mean: MeanTensor) -> ImageTensor:
        "Injects the new mean into the image."
        raise NotImplementedError


@dataclass
class AdaINInjection(Injection):
    """
    Implements oridinary AdaIN injection: a channelwise affine transform.
    (Adaptive Instance Normalization)
    In other words the scale/mean vector components simply scale/shift their respective channel.
    """

    def num_required_scale_vectors(self, num_image_channels: int) -> int:
        return 1

    def num_required_scale_features(self, num_image_channels: int) -> int:
        return num_image_channels

    def rescale(self, image: ImageTensor, scale: ScaleTensor) -> ImageTensor:
        self._validate_scale_shape(image, scale)
        return image * scale.squeeze(1).unsqueeze(2).unsqueeze(3)

    def recenter(self, image: ImageTensor, mean: MeanTensor) -> ImageTensor:
        return image + mean.squeeze(1).unsqueeze(2).unsqueeze(3)


def _test_adain_required_shape():
    injection = AdaINInjection()
    num_channels = random.randint(2, 10)

    assert injection.num_required_scale_vectors(num_channels) == 1
    assert injection.num_required_scale_features(num_channels) == num_channels


def _test_adain_injection():
    injection = AdaINInjection()
    image = torch.randn(2, 3, 4, 4)

    "The scale is injected correctly"
    scale = torch.randn(2, 1, 3)
    assert image.allclose(injection.rescale(image, scale) / scale.reshape(2, 3, 1, 1))

    "The mean is injected correctly"
    mean = torch.randn(2, 1, 3)
    assert image.allclose(injection.recenter(image, mean) - mean.reshape(2, 3, 1, 1))


@dataclass
class AdaConvInjection(Injection):
    """
    Implements AdaConv injection: channels are passed through a k by k convolution.
    (Adaptive Convolution)
    Here each scale vector defines a single k by k convolution filer, so we need as many vectors
    as there are channels in the image.
    The use of mean vectors is unchanged; conceptually they become the convolution's bias.
    """

    kernel_size: int = 1

    def __post_init__(self):
        assert (
            self.kernel_size % 2 == 1
        )  # * Kernel sizes must be odd in order for padding to preserve the shape

    def num_required_scale_vectors(self, num_image_channels: int) -> int:
        return num_image_channels

    def num_required_scale_features(self, num_image_channels: int) -> int:
        return num_image_channels * self.kernel_size ** 2

    def rescale(self, image: ImageTensor, scale: ScaleTensor) -> ImageTensor:
        """
        Uses convolution groups to perform a different convolution for each element in the batch.
        This means the channels of all batch elements are concatenated together, and the convolution
        is performed using a batch size of 1. To compensate, the convolution weights are divided into subgroups (one per batch element), where each group only affects the channels of its corresponding batch element.
        """
        self._validate_scale_shape(image, scale)

        B, C, H, W = image.shape

        def batch_to_conv_groups(x):
            return x.reshape(1, B * C, H, W)

        def conv_groups_to_batch(x):
            return x.reshape(*image.shape)

        return conv_groups_to_batch(
            F.conv2d(
                input=nn.ReflectionPad2d(self.kernel_size // 2)(
                    batch_to_conv_groups(image)
                ),
                weight=scale.reshape(
                    B * C,  # @ == grouped conv out_channels
                    C,  # @ == grouped conv in_channels (which is B * C) / num_conv_groups (which is B)
                    self.kernel_size,
                    self.kernel_size,
                )
                / math.sqrt(C),
                groups=B,
            )
        )

    recenter = AdaINInjection.recenter


def _test_adaconv_required_shape():
    for kernel_size in [1, 3, 5]:
        injection = AdaConvInjection(kernel_size=kernel_size)
        num_channels = random.randint(2, 10)

        assert injection.num_required_scale_vectors(num_channels) == num_channels
        assert (
            injection.num_required_scale_features(num_channels)
            == num_channels * kernel_size ** 2
        )


def _test_adaconv_injection():
    B, C, R = 2, 4, 6

    image = torch.randn(B, C, R, R)

    for kernel_size in [1, 3, 5]:
        injection = AdaConvInjection(kernel_size=kernel_size)

        "The scale is injected correctly"
        scale = torch.randn(B, C, C * kernel_size ** 2)
        if kernel_size == 1:
            # * Testing is easy in the 1x1 case
            assert image.allclose(
                scale.reshape(B, C, C)
                .inverse()
                .bmm(injection.rescale(image, scale).reshape(B, C, -1))
                .reshape(image.shape),
                atol=1e-3,
            )
        else:
            # * The other cases are hard; just test the shape for now
            assert injection.rescale(image, scale).shape == image.shape  # todo

        "The mean is injected correctly"
        mean = torch.randn(B, 1, C)
        assert image.allclose(
            injection.recenter(image, mean) - mean.reshape(B, C, 1, 1), atol=1e-3
        )


def _test_adaconv_maps_each_scale_vector_to_an_individual_conv_filter():
    B, C, R = 2, 4, 6

    image = torch.randn(B, C, R, R)

    for kernel_size in [1, 3, 5]:
        injection = AdaConvInjection(kernel_size=kernel_size)

        scale = torch.randn(B, C, C * kernel_size ** 2, requires_grad=True)

        for i in range(C):
            scale_grad = torch.autograd.grad(
                injection.rescale(image, scale)[:, i].sum(), scale
            )[0]

            """
            The ith channel of the output comes from the ith conv filter.
            These assertions verify that it is only a function of the ith style vector.
            """
            assert (scale_grad[:, :i, :] == 0).all(), scale_grad[:, :i, :]
            assert (scale_grad[:, i, :] != 0).all(), scale_grad[:, :i, :]
            assert (scale_grad[:, i + 1 :, :] == 0).all(), scale_grad[:, i + 1 :, :]
