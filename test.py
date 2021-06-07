import torch
from model.networks import *

x = torch.ones((64, 3, 64, 64))
x_coarse = CoarseGenerator(x, 32)
x_fine = FineGenerator(x_coarse)