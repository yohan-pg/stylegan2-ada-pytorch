import copy
import os

from time import perf_counter
import sys
import torch.optim as optim
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
from xml.dom import minidom
import shutil

from torchvision.utils import save_image, make_grid
from interpolator import interpolate_images
from training.networks import normalize_2nd_moment

from abc import ABC, abstractmethod, abstractstaticmethod, abstractclassmethod
from dataclasses import dataclass, field

from typing import Optional, Type, List, final, Tuple, Callable, Iterator, Iterable, Dict, ClassVar
import matplotlib.pyplot as plt 

from torchvision.io import write_video

ImageTensor = torch.Tensor # [B, C, H, W] with data between 0 and 1

class ToStyles(ABC, torch.nn.Module):
    @abstractmethod
    def to_styles(self):
        raise NotImplementedError


Styles = torch.Tensor

def dbg(x):
    print(x)
    return x

def imview(image):
    save_image(image, "tmp/tmp.png")
    os.system("code tmp/tmp.png")
