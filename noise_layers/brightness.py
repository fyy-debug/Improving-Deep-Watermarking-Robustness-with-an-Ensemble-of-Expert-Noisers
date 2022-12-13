import torch
import torch.nn as nn
import torchvision.transforms as transforms


class Brightness(nn.Module):
    def __init__(self, ratio=1.5):
        super(Brightness, self).__init__()
        self.brightness = transforms.ColorJitter(brightness=[ratio, ratio])

    def forward(self, noised_and_cover):
        noised_and_cover[0] = (noised_and_cover[0] + 1.0) / 2.0
        noised_and_cover[0] = self.brightness(noised_and_cover[0])
        noised_and_cover[0] = 2.0 * noised_and_cover[0] - 1.0
        return noised_and_cover
