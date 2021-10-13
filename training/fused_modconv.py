 #!
    # Execute by scaling the activations before and after the convolution.
    # if not fused_modconv:
    #     if styles.ndim == 3:
    #         S = styles[:, :x.shape[1], :x.shape[1]]
    #         # S = (S - 1) / math.sqrt(512) + 1
    #         # + torch.eye(x.shape[1]).unsqueeze(0).to(styles.device) / math.sqrt(2)
    #         # S = S * (S.square().sum(dim=[2], keepdim=True) + 1e-8).rsqrt() #! no normalization
    #         x = (S @ x.reshape(-1, x.shape[1], x.shape[2] * x.shape[3])).reshape(x.shape)
    #     else:
    #         x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
    #     x = conv2d_resample.conv2d_resample(
    #         x=x,
    #         w=weight.to(x.dtype),
    #         f=resample_filter,
    #         up=up,
    #         down=down,
    #         padding=padding,
    #         flip_weight=flip_weight,
    #     )
    #     x = torch.nn.InstanceNorm2d(x.shape[1])(x) #!
    #     if demodulate and noise is not None:
    #         x = fma.fma(
    #             x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype)
    #         )
    #     elif demodulate:
    #         x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
    #     elif noise is not None:
    #         x = x.add_(noise.to(x.dtype))
    #     return x
