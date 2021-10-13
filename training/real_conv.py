

@misc.profiled_function
def modulated_conv2d(
    x,  # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,  # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,  # Modulation coefficients of shape [batch_size, in_channels].
    noise=None,  # Optional noise tensor to add to the output activations.
    up=1,  # Integer upsampling factor.
    down=1,  # Integer downsampling factor.
    padding=0,  # Padding with respect to the upsampled image.
    resample_filter=None,  # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate=True,  # Apply weight demodulation?
    flip_weight=True,  # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv=True,  # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw])  # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None])  # [NIHW]
    if styles.ndim == 2:
        misc.assert_shape(styles, [batch_size, in_channels])  # [NI]
    else:
        assert styles.ndim == 3
        misc.assert_shape(styles, [batch_size, 512, in_channels])  # [NI]

    # Calculate per-sample weights and demodulation coefficients.
    if styles.ndim == 3:
        S = styles[:, :x.shape[1], :x.shape[1]] + torch.eye(x.shape[1]).to(styles.device)
        # S = (S - 1) / math.sqrt(512) + 1
        # + torch.eye(x.shape[1]).unsqueeze(0).to(styles.device) / math.sqrt(2)
        # S = S * (S.square().sum(dim=[2], keepdim=True) + 1e-8).rsqrt() #!! no normalization
        x = (S @ x.reshape(-1, x.shape[1], x.shape[2] * x.shape[3])).reshape(x.shape)
    else:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
    
    x = conv2d_resample.conv2d_resample(
        x=x,
        w=weight.to(x.dtype),
        f=resample_filter,
        up=up,
        down=down,
        padding=padding,
        flip_weight=flip_weight,
    )
    x = torch.nn.InstanceNorm2d(x.shape[1])(x)
    if noise is not None:
        x = x.add_(noise.to(x.dtype))
    return x
