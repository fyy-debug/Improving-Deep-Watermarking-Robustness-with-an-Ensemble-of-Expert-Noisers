import torch.nn as nn
import math
import torch.nn.init as init
from torch.nn.utils.spectral_norm import spectral_norm
from model.conv_bn_relu import ConvBNRelu

class Discriminator(nn.Module):
    """
    Discriminator network. Receives an image and has to figure out whether it has a watermark inserted into it, or not.
    """
    def __init__(self, FLAGS):
        super(Discriminator, self).__init__()

        layers = [ConvBNRelu(3, FLAGS.discriminator_channels)]
        for _ in range(FLAGS.discriminator_blocks-1):
            layers.append(ConvBNRelu(FLAGS.discriminator_channels, FLAGS.discriminator_channels))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.before_linear = nn.Sequential(*layers)
        self.linear = nn.Linear(FLAGS.discriminator_channels, 1)
        self.initialize()

    def initialize(self):
        for m in self.before_linear.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, math.sqrt(2))
                init.zeros_(m.bias)
                spectral_norm(m)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                spectral_norm(m)

    def forward(self, image):
        X = self.before_linear(image)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        X.squeeze_(3).squeeze_(2)
        X = self.linear(X)
        # X = torch.sigmoid(X)
        return X
