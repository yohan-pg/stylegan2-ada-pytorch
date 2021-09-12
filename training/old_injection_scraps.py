# s = (styles[:, :w.shape[2], :w.shape[2]] - 1) / math.sqrt(512) + torch.eye(w.shape[2]).to(w.device).unsqueeze(0) / math.sqrt(2)
            # x = s.bmm(x.reshape(*x.shape[0:2], -1)).reshape(x.shape)

            # --------------
            # def flip(k):
            #     return torch.flip(k, (2, 3)).transpose(0, 1)

            # def crosscorr(x, w, *args, **kwargs):
            #     return torch.nn.functional.conv2d(x, w, *args, **kwargs)

            # s = styles[:, :w.shape[1], :w.shape[0]].unsqueeze(3).unsqueeze(4)
            # w = torch.stack([
            #     flip(crosscorr(flip(s), w[0], padding=3-1)) for s in s
            # ])
            
            # s = torch.eye(w.shape[2]).to(s.device).unsqueeze(0).unsqueeze(3).unsqueeze(4).repeat(s.shape[0], 1, 1, 1, 1)

            # s = styles[:, :w.shape[2], :w.shape[2]].unsqueeze(3).unsqueeze(4) #* when using a conv, we want to preserve the shape
            # s = ((s - 1) / math.sqrt(512)) * 0.0 + torch.eye(w.shape[2]).to(s.device).unsqueeze(0).unsqueeze(3).unsqueeze(4)
            # s = torch.eye(w.shape[2]).to(s.device).unsqueeze(0).unsqueeze(3).unsqueeze(4).repeat()
            # dcoefs_s = (s.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
            # s = s * dcoefs_s.reshape(batch_size, -1, 1, 1, 1)  # [NOIkk]
            # w = w.repeat(s.shape[0], 1, 1, 1, 1)


# if styles.ndim == 3: # * We are using adaconv
    #     s = s.reshape(-1, in_channels, 1, 1)
    #     x2 = conv2d_resample.conv2d_resample(
    #         x=x,
    #         w=s.to(x.dtype),
    #         f=resample_filter,
    #         groups=batch_size,
    #         flip_weight=True, #* matches nn.conv2d
    #     )
    #     print(x - x2)
    #     x = x2