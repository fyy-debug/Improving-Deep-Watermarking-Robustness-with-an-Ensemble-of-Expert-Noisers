import torch
import torch.nn as nn
import torchvision.transforms as transforms


class Sat(nn.Module):
    def __init__(self, ratio=15.0):
        super(Sat, self).__init__()
        self.sat = transforms.ColorJitter(saturation=ratio)

    def forward(self, noised_and_cover):
        noised_and_cover[0] = (noised_and_cover[0] + 1.0) / 2.0
        noised_and_cover[0] = self.sat(noised_and_cover[0])
        noised_and_cover[0] = 2.0 * noised_and_cover[0] - 1.0
        return noised_and_cover
