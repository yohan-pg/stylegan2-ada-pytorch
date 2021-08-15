# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import training.networks as networks
from torch_utils.misc import copy_params_and_buffers

import legacy


@torch.no_grad()
def generate_images(
    seeds: Optional[List[int]] = range(10),
    network_pkl: str = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl",
    truncation_psi: float = 1.0,
    noise_mode: str = "const",
    outdir: str = "out",
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device("cuda")
    with dnnlib.util.open_url(network_pkl) as f:
        old_G = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore

    G = networks.Generator(*old_G.init_args, **old_G.init_kwargs, use_adaconv=False).to(
        device
    )
    copy_params_and_buffers(old_G, G, require_all=True)

    os.makedirs(outdir, exist_ok=True)

    if seeds is not None:
        print("warn: --seeds is ignored when using --projected-w")
    print(f"Generating images")

    edit = 0.0 * torch.randn(1, 1, 512).to(device)

    for seed_idx, seed in enumerate(seeds):
        print("Generating image for seed %d (%d/%d) ..." % (seed, seed_idx, len(seeds)))

        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        c = torch.zeros(z.shape).to(device)

        ws = G.mapping(z, c, truncation_psi=truncation_psi)
        assert ws.shape == (1, G.num_ws, G.w_dim)

        img1 = G.synthesis(ws, noise_mode=noise_mode)
        img1 = (img1.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        img2 = G.synthesis(ws * 0.3, noise_mode=noise_mode)
        img2 = (img2.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        PIL.Image.fromarray(torch.hstack((img1, img2))[0].cpu().numpy(), "RGB").save(
            f"{outdir}/offset{seed_idx:02d}.png"
        )
    return


if __name__ == "__main__":
    generate_images()
