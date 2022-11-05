import torch.nn as nn
import torchvision.transforms as T


class Hue(nn.Module):
    """
    Identity-mapping noise layer. Does not change the image
    """
    def __init__(self, ratio=0.2):
        super(Hue, self).__init__()
        self.hue = T.ColorJitter(hue=ratio)

    def forward(self, noised_and_cover):
        noised_and_cover[0] = (noised_and_cover[0]+1.0)/ 2.0
        noised_and_cover[0] = self.hue(noised_and_cover[0])
        noised_and_cover[0] = 2.0 * noised_and_cover[0] - 1.0
        return noised_and_cover
