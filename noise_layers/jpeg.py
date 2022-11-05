from PIL import Image
import io
import torchvision.transforms as T
import torch
import torchvision
import torch.nn as nn

class JPEG(nn.Module):
    """
    Identity-mapping noise layer. Does not change the image
    """
    def __init__(self,quality=50):
        super(JPEG, self).__init__()
        self.quality = quality


    def forward(self, noised_and_cover):
        noised_and_cover[0] = (noised_and_cover[0] + 1.0) / 2.0
        noised_and_cover[0] = torch.clamp(noised_and_cover[0], min=0.0, max=1.0)
        noised_and_cover[0] = self.encoding_quality(noised_and_cover[0],quality=self.quality)
        noised_and_cover[0] = 2.0 * noised_and_cover[0] - 1.0
        return noised_and_cover

    def encoding_quality(self,images: Image.Image,quality:int):

        assert 0 <= quality <= 100, "'quality' must be a value in the range [0, 100]"
        #assert isinstance(images, Image.Image), "Expected type PIL.Image.Image for variable 'image'"

        batch_aug_image = []
        for image in images:
            image = T.ToPILImage()(image)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality)
            aug_image = Image.open(buffer)
            aug_image = T.ToTensor()(aug_image)
            batch_aug_image.append(aug_image.unsqueeze(0))

        batch_aug_image = torch.cat(batch_aug_image,dim = 0).to(images.device)
        #print("batch_aug:",batch_aug_image.shape)
        #torchvision.utils.save_image(images, "ori.png")
        #torchvision.utils.save_image(batch_aug_image, "jpeg_image.png")
        return batch_aug_image

'''
if __name__ == '__main__':
    img = torchvision.io.read_image("/home/yeochengyu/PycharmProjects/pythonProject/BSC_FLIP/000000521601.jpg")/255.
    print(img)
    #img = torch.rand(3,128,128)
    aug_img = encoding_quality(img,quality=100)
    print(aug_img)
    torchvision.utils.save_image(aug_img,"jpeg_image.png")
    print("aug:",aug_img.shape)
'''