import torch
import torch.nn as nn
import torchvision.transforms as transforms

class Blur(nn.Module):
    def __init__(self, ratio=1.0):
        super(Blur, self).__init__()
        self.blur = transforms.GaussianBlur(3,ratio)

    def forward(self, noised_and_cover):
        noised_and_cover[0] = self.blur(noised_and_cover[0])
        return noised_and_cover
