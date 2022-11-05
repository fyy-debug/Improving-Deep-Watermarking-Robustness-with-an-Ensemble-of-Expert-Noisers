import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self):
        super(Generator, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 3, 3, stride=1, padding=1),
        )

    def forward(self, enc_image):
        return self.conv_layers(enc_image)



