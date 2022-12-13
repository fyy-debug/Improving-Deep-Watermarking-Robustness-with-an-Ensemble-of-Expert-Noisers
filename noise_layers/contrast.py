import torch
import torch.nn as nn
import torchvision.transforms as transforms


class Contrast(nn.Module):
    def __init__(self, ratio=2.0):
        super(Contrast, self).__init__()
        self.contrast = transforms.ColorJitter(contrast=[ratio, ratio])

    def forward(self, noised_and_cover):
        noised_and_cover[0] = (noised_and_cover[0] + 1.0) / 2.0
        noised_and_cover[0] = self.contrast(noised_and_cover[0])
        noised_and_cover[0] = 2.0 * noised_and_cover[0] - 1.0
        return noised_and_cover
