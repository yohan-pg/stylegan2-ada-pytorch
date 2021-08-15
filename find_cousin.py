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

    def sample_G(ws=None):
        if ws is None:
            z = torch.randn(1, G.z_dim).to(device)
            c = torch.zeros(z.shape).to(device)
            ws = G.mapping(z, c, truncation_psi=truncation_psi)
        assert ws.shape == (1, G.num_ws, G.w_dim)
        img = G.synthesis(ws, noise_mode=noise_mode)
        return img, ws

    def convert(im):
        return (im.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

    person, ws_person = sample_G()
    criterion = torch.nn.MSELoss()

    best = (*sample_G(), float("inf"))

    for i in range(100000):
        candidate, ws_candidate = sample_G()
        loss = criterion(candidate, person).item()
        if loss < best[2]:
            print("NEW MATCH", loss)
            best = (candidate, ws_candidate, loss)

            cousin, ws_cousin, _ = best
            inbetween, ws_inbetween = sample_G((ws_person + ws_cousin) / 2)
            PIL.Image.fromarray(
                torch.hstack((convert(person), convert(inbetween), convert(cousin)))[0]
                .cpu()
                .numpy(),
                "RGB",
            ).save(f"{outdir}/closest_match.png")

    return


if __name__ == "__main__":
    generate_images()
