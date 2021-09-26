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

import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy
import pickle


def convert(im):
    return (im.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)


def interpolate_images(
    w1_path: str,
    w2_path: str,
    network_pkl: str,
    outdir: str = "./out/interpolate",
    num_steps: int = 7,
):
    device = torch.device("cuda")

    print("Interpolating:", w1_path, w2_path)

    print("Loading style vectors...")
    w1 = torch.tensor(np.load(w1_path + "/projected_w.npz")["w"], device=device)
    w2 = torch.tensor(np.load(w2_path + "/projected_w.npz")["w"], device=device)

    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Synthesize the result of a W projection.
    print(f"Interpolating images ")

    imgs = []
    for i in range(7):
        print(i)
        alpha = i / (num_steps - 1)
        img = G.synthesis((1.0 - alpha) * w1 + alpha * w2, noise_mode="const")
        imgs.append(img)
    print()

    name_1 = w1_path.split("/")[-1]
    name_2 = w2_path.split("/")[-1]
    mosaic = torch.cat(imgs, dim=3)
    PIL.Image.fromarray(convert(mosaic)[0].cpu().numpy(), "RGB").save(
        f"{outdir}/mix_{name_1}_{name_2}.png"
    )

    # Synthesize the result of a Z projection.
    print(f"Interpolating images on Z")

    imgs = []
    for i in range(7):
        print(i)
        alpha = i / (num_steps - 1)
        img = G.synthesis(G.mapping((1.0 - alpha) * w1[:, :G.num_required_vectors()] + alpha * w2[:, :G.num_required_vectors()], None), noise_mode="const")
        imgs.append(img)
    print()

    name_1 = w1_path.split("/")[-1]
    name_2 = w2_path.split("/")[-1]
    mosaic = torch.cat(imgs, dim=3)
    PIL.Image.fromarray(convert(mosaic)[0].cpu().numpy(), "RGB").save(
        f"{outdir}/mix_z_{name_1}_{name_2}.png"
    )


# ----------------------------------------------------------------------------


# if __name__ == "__main__":
#     styles = ["./out/church1", "./out/church2"]
#     for i, x in enumerate(styles):
#         for y in styles[i + 1 :]:
#             interpolate_images(x, y)
#             print()
# if __name__ == "__main__":
#     styles = ["./out/real1a", "./out/real1b"]
#     for i, x in enumerate(styles):
#         for y in styles[i + 1 :]:
#             interpolate_images(x, y, "./pretrained/stylegan2-ffhq-config-f.pkl")
#             print()

if __name__ == "__main__":
    import sys 
    styles = sys.argv[2:]
    for i, x in enumerate(styles):
        for y in styles[i + 1 :]:
            interpolate_images(x, y, sys.argv[1])
            print()

# ----------------------------------------------------------------------------
