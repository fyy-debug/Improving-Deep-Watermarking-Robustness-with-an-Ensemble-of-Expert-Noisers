import torch.nn as nn
from model.conv_bn_relu import ConvBNRelu
import torch

class Decoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self, FLAGS):
        super(Decoder, self).__init__()
        self.channels = FLAGS.decoder_channels

        layers = [ConvBNRelu(3, self.channels)]
        layers.append(ConvBNRelu(self.channels, self.channels, stride=1))
        layers.append(ConvBNRelu(self.channels, self.channels, stride=1))
        layers.append(ConvBNRelu(self.channels, self.channels, stride=1))
        layers.append(ConvBNRelu(self.channels, self.channels, stride=1))
        layers.append(ConvBNRelu(self.channels, self.channels, stride=2))
        layers.append(ConvBNRelu(self.channels, self.channels, stride=2))
        layers.append(ConvBNRelu(self.channels, FLAGS.redundant_length,stride=1))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(FLAGS.redundant_length, FLAGS.redundant_length)

    def forward(self, image_with_wm):
        #print("img:",image_with_wm.min(),image_with_wm.max())
        image_with_wm = (image_with_wm + 1.)/2.
        image_with_wm = torch.clamp(image_with_wm, min=0.0, max=1.0)
        x = self.layers(image_with_wm)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        x.squeeze_(3).squeeze_(2)
        x = self.linear(x)
        return x
