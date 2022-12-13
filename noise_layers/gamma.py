import torch
import torch.nn as nn
import torchvision.transforms as transforms

class Gamma(nn.Module):
    def __init__(self, gamma=5.0, gain=2):
        super(Gamma, self).__init__()
        self.gamma = gamma
        self.gain = gain

    def forward(self, noised_and_cover):
        noised_and_cover[0] = transforms.functional.adjust_gamma(noised_and_cover[0], gamma=self.gamma, gain=self.gain)
        return noised_and_cover
