import torch 
from dataclasses import dataclass, field
import math 
import torch.nn as nn
import torch.nn.functional as F

ImageTensor = torch.Tensor #TensorType["batch", "image_channels", "rows", "columns"]
StyleTensor = torch.Tensor #TensorType["batch", "vectors", "style_features"]

MeanTensor = torch.Tensor #TensorType["batch", 1, "image_channels"]
ScaleTensor = torch.Tensor #TensorType["batch", "vectors", "scale_features"]