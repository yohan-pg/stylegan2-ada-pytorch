import copy
import os

from time import perf_counter

import tqdm
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

from abc import ABC, abstractmethod, abstractstaticmethod, abstractclassmethod
from dataclasses import dataclass

from typing import Optional, Type, List, final, Tuple, Callable


ImageTensor = torch.Tensor # [B, C, H, W] with data between 0 and 1

class ToStyles(ABC, torch.nn.Module):
    @abstractmethod
    def to_styles(self):
        raise NotImplementedError


Styles = torch.Tensor