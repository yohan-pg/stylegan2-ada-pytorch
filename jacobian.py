# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import os
from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import dnnlib
import legacy

from torchvision.utils import save_image
from interpolator import interpolate_images

import matplotlib.pyplot as plt
from training import networks
from tqdm import tqdm

# todo optimize code, understand the jvp trick

def measure_jacobian(
    network_pkl: str,
    untrained=False, 
    seed: int=0,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device("cuda")
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)["G_ema"].requires_grad_(False).to(device)  # type: ignore

    if untrained:
        init_kwargs = {**G.init_kwargs}
        # init_kwargs['mapping_kwargs'] = {**init_kwargs['mapping_kwargs']}
        # init_kwargs['mapping_kwargs']['num_layers'] = 8 
        # init_kwargs['mapping_kwargs']['activation'] = "tanh" 
        # init_kwargs['mapping_kwargs']['orthogonal_init'] = True 
        G = networks.Generator(*G.init_args, **init_kwargs).to(device)
    
    model = G.synthesis 

    D = 512

    z = torch.randn(D).cuda()
    def f(z):
        if model.__class__.__name__ == "SynthesisNetwork":
            return model(z.unsqueeze(0).repeat(1, G.mapping.num_ws, 1))
        else:
            return model(z.unsqueeze(0), None, noise_mode="const")
    x = f(z)
    print("Computing jacobian...")

    derivatives = []
    for i in tqdm(range(D)):
        seed = torch.zeros_like(z)
        seed[i] = 1.0
        derivatives.append(torch.autograd.functional.jvp(f, z, seed)[1])

    J = torch.stack(derivatives).reshape(D, -1)
    J /= J.max()
    M = J @ J.transpose(0, 1)

    sigmas = torch.svd(M).S
    sigmas /= sigmas[0].item()
    print(sigmas)
    print(sigmas.max(), sigmas.min())
    
    class make_plot:
        plt.clf()
        plt.title("Singular values of JJ^t for a latent z, divided by the largest value")
        plt.plot(sigmas.cpu())
        plt.ylim(0, 1.05)
        plt.savefig("sigmas.png")
    

if __name__ == "__main__":
    pickle = "pretrained/adain-dropout.pkl"
    measure_jacobian(pickle)
