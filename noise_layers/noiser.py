import numpy as np
import torch.nn as nn
from noise_layers.identity import Identity
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.dropout import Dropout
from noise_layers.cropout import Cropout
from noise_layers.crop import Crop
from noise_layers.resize import Resize
from noise_layers.hue import Hue
from noise_layers.gaussian_noise import Gaussian_Noise
from noise_layers.sat import Sat
from noise_layers.blur import Blur
from noise_layers.jpeg import JPEG


class Noiser(nn.Module):
    """
    This module allows to combine different noise layers into a sequential noise module. The
    configuration and the sequence of the noise layers is controlled by the noise_config parameter.
    """
    def __init__(self,device):
        super(Noiser, self).__init__()
        self.noise_layers = [Identity()]
        self.noise_layers.append(Dropout([0.3, 0.3]))
        self.noise_layers.append(Cropout([0.25, 0.35], [0.25, 0.35]))
        self.noise_layers.append(JPEG(quality=50))
        self.noise_layers.append(Crop([0.19, 0.19], [0.19, 0.19]))
        self.noise_layers.append(Resize([0.7, 0.7]))
        self.noise_layers.append(Hue(0.2))
        self.noise_layers.append(Sat())
        self.noise_layers.append(Blur())
        self.noise_layers.append(Gaussian_Noise())

    def forward(self, encoded_and_cover):
        random_noise_layer = np.random.choice(self.noise_layers[:4], 1)[0]
        return random_noise_layer(encoded_and_cover)

