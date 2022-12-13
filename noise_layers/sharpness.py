import torch
import torch.nn as nn
import torchvision.transforms as transforms

class Sharp(nn.Module):
    def __init__(self, ratio=3.0):
        super(Sharp, self).__init__()
        self.ratio = ratio

    def forward(self, noised_and_cover):
        noised_and_cover[0] = transforms.functional.adjust_sharpness(noised_and_cover[0], self.ratio)
        return noised_and_cover
