import copy
import os

from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import dnnlib
import legacy

from torchvision.utils import save_image
from interpolator import interpolate_images
from training.networks import normalize_2nd_moment

from abc import ABC, abstractmethod
from dataclasses import dataclass


class ToStyles(ABC, torch.nn.Module):
    @abstractmethod
    def variable_to_styles(self):
        raise NotImplementedError


Styles = torch.Tensor