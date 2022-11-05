import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder

class EncoderDecoder(nn.Module):
    """
    Combines Encoder->Noiser->Decoder into single pipeline.
    The input is the cover image and the watermark message. The module inserts the watermark into the image
    (obtaining encoded_image), then applies Noise layers (obtaining noised_image), then passes the noised_image
    to the Decoder which tries to recover the watermark (called decoded_message). The module outputs
    a three-tuple: (encoded_image, noised_image, decoded_message)
    """
    def __init__(self, FLAGS):

        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(FLAGS)
        self.decoder = Decoder(FLAGS)

    def forward(self, image, message, generator, necst, identity=False, noiser=None):

        if identity is False:
            redundant_message = necst.encode(message)
            encoded_image = self.encoder(image, redundant_message)
            adv_image = generator(encoded_image)
            adv_redundant_decoded_message = self.decoder(adv_image)
            redundant_decoded_message = self.decoder(encoded_image)
            return encoded_image, adv_image, redundant_decoded_message, adv_redundant_decoded_message
        else:
            encoded_image = self.encoder(image, message)
            if noiser is not None:
                noised_and_cover = noiser([encoded_image, image])
                noised_image = noised_and_cover[0]
                decoded_message = self.decoder(noised_image)
            else:
                decoded_message = self.decoder(encoded_image)

            return encoded_image, decoded_message

