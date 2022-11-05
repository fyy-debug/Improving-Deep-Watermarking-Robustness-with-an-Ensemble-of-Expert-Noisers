import torch
from torch import nn
import torchjpeg.dct as DCT
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

# helpers
def rgb2yuv(image_rgb, image_yuv_out):
    """ Transform the image from rgb to yuv """
    image_yuv_out[:, 0, :, :] = 0.299 * image_rgb[:, 0, :, :].clone() + 0.587 * image_rgb[:, 1, :, :].clone() + 0.114 * image_rgb[:, 2, :, :].clone()
    image_yuv_out[:, 1, :, :] = -0.14713 * image_rgb[:, 0, :, :].clone() + -0.28886 * image_rgb[:, 1, :, :].clone() + 0.436 * image_rgb[:, 2, :, :].clone()
    image_yuv_out[:, 2, :, :] = 0.615 * image_rgb[:, 0, :, :].clone() + -0.51499 * image_rgb[:, 1, :, :].clone() + -0.10001 * image_rgb[:, 2, :, :].clone()


def yuv2rgb(image_yuv, image_rgb_out):
    """ Transform the image from yuv to rgb """
    image_rgb_out[:, 0, :, :] = image_yuv[:, 0, :, :].clone() + 1.13983 * image_yuv[:, 2, :, :].clone()
    image_rgb_out[:, 1, :, :] = image_yuv[:, 0, :, :].clone() + -0.39465 * image_yuv[:, 1, :, :].clone() + -0.58060 * image_yuv[:, 2, :, :].clone()
    image_rgb_out[:, 2, :, :] = image_yuv[:, 0, :, :].clone() + 2.03211 * image_yuv[:, 1, :, :].clone()

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
    return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0.,
                 emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.return_patch_embedding = nn.Sequential(
            nn.Linear(dim, patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=patch_height, p2=patch_width,
                      w=image_height // patch_height),
            nn.Conv2d(3, 3, 3, padding=1),
        )

        self.pos_embedding = nn.Embedding(3 * 8 * 8, dim)
        self.dropout = nn.Dropout(emb_dropout)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.image_size = image_size
        self.jpeg_mask = None
        yuv_keep_weights = (25,9,9)
        self.yuv_keep_weighs = yuv_keep_weights
        self.create_mask((1000, 1000))


    def create_mask(self, requested_shape):
        if self.jpeg_mask is None or requested_shape > self.jpeg_mask.shape[1:]:
            self.jpeg_mask = torch.empty((3,) + requested_shape, device=self.device)
            for channel, weights_to_keep in enumerate(self.yuv_keep_weighs):
                mask = torch.from_numpy(self.get_jpeg_yuv_filter_mask(requested_shape, 8, weights_to_keep))
                self.jpeg_mask[channel] = mask

    def get_jpeg_yuv_filter_mask(self,image_shape: tuple, window_size: int, keep_count: int):
        mask = np.zeros((window_size, window_size), dtype=np.uint8)

        index_order = sorted(((x, y) for x in range(window_size) for y in range(window_size)),
                             key=lambda p: (p[0] + p[1], -p[1] if (p[0] + p[1]) % 2 else p[1]))

        for i, j in index_order[0:keep_count]:
            mask[i, j] = 1

        return np.tile(mask, (int(np.ceil(image_shape[0] / window_size)),
                              int(np.ceil(image_shape[1] / window_size))))[0: image_shape[0], 0: image_shape[1]]

    def get_mask(self, image_shape):
        if self.jpeg_mask.shape < image_shape:
            self.create_mask(image_shape)
        # return the correct slice of it
        return self.jpeg_mask[:, :image_shape[1], :image_shape[2]].clone()

    def jpegmask(self, img):
        N, C, H, W = img.shape
        B = H // 8
        image_yuv = torch.empty_like(img)
        rgb2yuv(img, image_yuv)
        pos_emb = self.pos_embedding(torch.arange(3 * 8 * 8).unsqueeze(0).to(img.device))
        block_dct = DCT.batch_dct(image_yuv) # [batch_size,3, H, W]
        mask = self.get_mask(block_dct.shape[1:])
        block_dct = torch.mul(block_dct, mask) # [batch_size,3, H, W]
        block_dct = DCT.blockify(block_dct, 8)
        x = block_dct
        x = DCT.block_idct(x)
        x = DCT.deblockify(x, size=(H, W))
        adv_image = torch.empty_like(x)
        yuv2rgb(x, adv_image)
        return adv_image

    def forward(self, img):
        N, C, H, W = img.shape
        B = H // 8
        image_yuv = torch.empty_like(img)
        rgb2yuv(img, image_yuv)
        #pos_emb = self.pos_embedding(torch.arange(3 * 8 * 8).unsqueeze(0).to(img.device))
        block_dct = DCT.batch_dct(image_yuv) # [batch_size,3, H, W]
        mask = self.get_mask(block_dct.shape[1:])
        block_dct = torch.mul(block_dct, mask) # [batch_size,3, H, W]
        block_dct = DCT.blockify(block_dct, 8)  # [batch_size,3, (H/8)*(W/8) , 8, 8]
        block_dct = torch.flatten(block_dct, start_dim=-2)  # [batch_size,3, (H/8)*(W/8) , 64]
        block_dct = block_dct.permute(0, 1, 3, 2)  # [batch_size,3, 64 , (H/8)*(W/8)]
        x = torch.flatten(block_dct, start_dim=1, end_dim=2)  # [batch_size,3 * 64 , (H/8)*(W/8)]
        #x += pos_emb
        x = self.dropout(x)
        x = self.transformer(x)  # [batch_size,3 * 64 , (H/8)*(W/8)]
        x = x.reshape(N, C, 8 * 8, B * B)
        x = x.permute(0, 1, 3, 2)  # [batch_size,3 ,(H/8)*(W/8), 64]
        x = x.reshape(-1, C, B * B, 8, 8)  # [batch_size,3 ,(H/8)*(W/8), 8,8]
        x = DCT.block_idct(x)
        x = DCT.deblockify(x, size=(H, W))
        adv_image = torch.empty_like(x)
        yuv2rgb(x, adv_image)
        return adv_image

'''
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net_Gen = ViT(
        image_size=128,
        patch_size=8,
        dim=256,
        depth=6,
        heads=12,
        mlp_dim=256,
        dropout=0.0,
        emb_dropout=0.0
    ).to(device)

    input = torch.rand(32, 3, 128, 128).to(device)
    output = net_Gen(input)
    print(output.shape)

    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = torch.arange(10).unsqueeze(0).to(device).repeat(5,1)
    #print(x)
    #x = x.permute(0,2,1)
    x = torch.flatten(x,start_dim=0)
    x = x.reshape(5,10)
    print(x)
'''
if __name__ == '__main__':
    import torchvision
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net_Gen = ViT(
        image_size=128,
        patch_size=8,
        dim=256,
        depth=6,
        heads=12,
        mlp_dim=256,
        dropout=0.0,
        emb_dropout=0.0
    ).to(device)
    img = torchvision.io.read_image("/home/yeochengyu/PycharmProjects/pythonProject/BSC_FLIP/000000521601.jpg")/255.
    img = img.unsqueeze(0).to(device)
    img = img[:,:,:480,:480]
    #print(img)
    #img = torch.rand(3,128,128)
    aug_img = net_Gen.jpegmask(img)
    #print(aug_img)
    torchvision.utils.save_image(img, "ori_image.png")
    torchvision.utils.save_image(aug_img,"jpeg_image.png")
    print("aug:",aug_img.shape)

