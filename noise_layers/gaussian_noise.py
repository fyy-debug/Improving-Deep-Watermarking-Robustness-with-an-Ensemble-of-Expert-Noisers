import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Gaussian_Noise(nn.Module):
    def __init__(self, mean=0, std=0.06):
        super(Gaussian_Noise, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, noised_and_cover):
        noise = torch.normal(torch.full_like(noised_and_cover[0], self.mean), torch.full_like(noised_and_cover[0], self.std))
        noised_and_cover[0] += noise
        return noised_and_cover