# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import click
import re
import json
import tempfile
import torch
import dnnlib

from training import training_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops

from inversion import *

METHOD = "adaconv"
G_PATH = f"pretrained/alpha-{METHOD}-002600.pkl"

if __name__ == "__main__":
    G = open_generator(G_PATH)
    criterion = VGGCriterion()

    target = open_target(G, "datasets/samples/cats/00000/img00000014.png")

    for i in range(1_000_000):

        def eval():
            optimizer.zero_grad()
            pred = variable.to_image()
            loss = criterion(downsample(pred), target_low_res)
            loss.backward()
            return loss

        loss = optimizer.step(eval)

        if i % 100 == 0:
            pred = variable.to_image()
            print(loss.item())
            save_image(torch.cat((pred, target)), "out/pulse_result.png")
            save_image(
                torch.cat((downsample(pred), target_low_res)), "out/pulse_optim.png"
            )
