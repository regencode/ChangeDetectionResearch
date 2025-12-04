import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import crop, resize
import einops as ein
import cv2 as cv
import numpy as np


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor, mask: torch.Tensor):
        if torch.rand(()) < self.p:
            # horizontal flip
            x1 = torch.flip(x1, dims=[-2])
            x2 = torch.flip(x2, dims=[-2])
            mask = torch.flip(mask, dims=[-2])

        return x1, x2, mask

class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor, mask: torch.Tensor):
        if torch.rand(()) < self.p:
            # vertical flip
            x1 = torch.flip(x1, dims=[-1])
            x2 = torch.flip(x2, dims=[-1])
            mask = torch.flip(mask, dims=[-1])

        return x1, x2, mask

class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor, mask: torch.Tensor):
        if torch.rand(()) < self.p:
            rand_flip = torch.randint(0, 3, (1,)).item()
            if rand_flip == 0:
                # horizontal flip
                x1 = torch.flip(x1, dims=[-2])
                x2 = torch.flip(x2, dims=[-2])
                mask = torch.flip(mask, dims=[-2])

            elif rand_flip == 1:
                # vertical flip
                x1 = torch.flip(x1, dims=[-1])
                x2 = torch.flip(x2, dims=[-1])
                mask = torch.flip(mask, dims=[-1])

            else:
                # horizontal + vertical flip
                x1 = torch.flip(x1, dims=[-2, -1])
                x2 = torch.flip(x2, dims=[-2, -1])
                mask = torch.flip(mask, dims=[-2, -1])

        return x1, x2, mask


class RandomScale:
    def __init__(self, min=0.8, max=1.2):
        self.min = min
        self.max = max
    def __call__(self, x1: torch.Tensor, x2: torch.Tensor, mask: torch.Tensor):
        assert len(x1.shape) == 3, f"x dims must be 3, got {len(x1.shape)}"
        min = self.min
        max = self.max
        C, H, W = x1.shape[-3:]
        rand_scale = torch.rand(()) * (max-min) + min
        new_H = int(H * rand_scale)
        new_W = int(W * rand_scale)

        new_x1 = F.interpolate(x1.unsqueeze(0), size=(new_H, new_W), mode="bilinear")
        new_x2 = F.interpolate(x2.unsqueeze(0), size=(new_H, new_W), mode="bilinear")
        new_mask = F.interpolate(mask.unsqueeze(0), size=(new_H, new_W), mode="nearest")
        return new_x1, new_x2, new_mask

class RandomResizedCropPair:
    def __init__(self, size, scale=(0.8,1.0), ratio=(1.0,1.0)):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, x1, x2, mask):
        i, j, h, w = T.RandomResizedCrop.get_params(
            torch.zeros_like(x1),
            scale=self.scale,
            ratio=self.ratio
        )
        x1 = crop(x1, i, j, h, w)
        x2 = crop(x2, i, j, h, w)
        mask = crop(mask, i, j, h, w)

        H_crop, W_crop = self.size
        x1 = resize(x1, (H_crop, W_crop))
        x2 = resize(x2, (H_crop, W_crop))
        mask = resize(mask, (H_crop, W_crop), interpolation=T.InterpolationMode.NEAREST)

        return x1, x2, mask

class RandomBlur:
    def __init__(self, p=0.5, kernel_size=3):
        self.blur = T.GaussianBlur(kernel_size)
        self.p = p

    def __call__(self, x1, x2, mask):
        if torch.rand(()) < self.p:
            x1 = self.blur(x1)
            x2 = self.blur(x2)
        return x1, x2, mask


class RandomColorJitter:
    def __init__(self, p=0.5):
        self.jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        self.p = p

    def __call__(self, x1, x2, mask):
        if torch.rand(()) < self.p:
            x1 = self.jitter(x1)
        if torch.rand(()) < self.p:
            x2 = self.jitter(x2)
        return x1, x2, mask


class PairCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __getitem__(self, i):
        return self.transforms[i]

    def __call__(self, x1, x2, mask):
        for t in self.transforms:
            x1, x2, mask = t(x1, x2, mask)
        return x1, x2, mask

def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.cpu().numpy()

def lanczos_resample_torch(x : torch.Tensor, intermediate_size):
    x_new = to_numpy(x)
    x_new = ein.rearrange(x_new, "c w h -> w h c")
    W, H, C = x_new.shape
    new_W, new_H = intermediate_size
    x_new = cv.resize(x_new, (new_W, new_H), interpolation=cv.INTER_LANCZOS4)
    x_new = cv.resize(x_new, (W, H), interpolation=cv.INTER_LANCZOS4)
    return ein.rearrange(torch.from_numpy(x_new), "w h c -> c w h")

class Interpolate:
    def __init__(self, size : tuple, mode="bilinear", upsample_mask=True):
        self.mode = mode
        self.upsample_mask = upsample_mask
        self.size = size

    def __call__(self, x1, x2, mask):
        if self.mode.lower() == "lanczos":
            x1 = lanczos_resample_torch(x1, self.size)
            x2 = lanczos_resample_torch(x2, self.size)
        else:
            C, W, H = x1.shape
            new_W = W*1.2
            new_H = H*1.2
            x1 = F.interpolate(x1.unsqueeze(0), size=(new_W, new_H), mode=self.mode, align_corners=False if self.mode == "bilinear" else None)
            x1 = F.interpolate(x1, size=(W, H), mode=self.mode).squeeze(0)

            x2 = F.interpolate(x2.unsqueeze(0), size=(new_W, new_H), mode=self.mode, align_corners=False if self.mode == "bilinear" else None)
            x2 = F.interpolate(x2, size=(W, H), mode=self.mode).squeeze(0)
        if self.upsample_mask:
            mask = F.interpolate(mask, size=self.size, mode="nearest")
        return x1, x2, mask

class Resample:
    def __init__(self, intermediate_size : tuple, mode="bilinear"):
        self.mode = mode
        self.size = intermediate_size

    def __call__(self, x1, x2, mask):
        C, W, H = x1.shape
        new_W, new_H = self.size
        if self.mode.lower() == "lanczos":
            x1 = lanczos_resample_torch(x1, self.size)
            x2 = lanczos_resample_torch(x2, self.size)
        else:
            x1 = F.interpolate(x1.unsqueeze(0), size=(new_H, new_W),
                               mode=self.mode,
                               align_corners=False if self.mode in ["bilinear", "bicubic"] else None)
            x1 = F.interpolate(x1, size=(H, W),
                           mode=self.mode,
                           align_corners=False if self.mode in ["bilinear", "bicubic"] else None).squeeze(0)

            x2 = F.interpolate(x2.unsqueeze(0), size=(new_H, new_W),
                           mode=self.mode,
                           align_corners=False if self.mode in ["bilinear", "bicubic"] else None)
            x2 = F.interpolate(x2, size=(H, W),
                           mode=self.mode,
                           align_corners=False if self.mode in ["bilinear", "bicubic"] else None).squeeze(0)

        return x1, x2, mask

class Sobel(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.sobel_kernel = nn.Conv2d(in_channels, 2, kernel_size=3, padding="same")

        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32)

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=torch.float32)

        with torch.no_grad():
            for c in range(in_channels):
                self.sobel_kernel.weight[0, c] = sobel_x
                self.sobel_kernel.weight[1, c] = sobel_y 
            self.sobel_kernel.bias.zero_()

            # Freeze parameters
            self.sobel_kernel.weight.requires_grad_(False)
            self.sobel_kernel.bias.requires_grad_(False)

    def forward(self, x):
        g = self.sobel_kernel(x)

        gx = g[:, 0]
        gy = g[:, 1]
        return torch.abs(gx) + torch.abs(gy)



