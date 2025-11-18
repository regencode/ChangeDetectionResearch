import torch
from torch import nn
import time
import torch.nn.functional as F
import einops as ein
from VMamba import *
from ChangeFormerDecoder import *

start_time = time.time()
class CustomModel(nn.Module):
    def __init__(self, in_channels, num_classes,
                 C=[32, 64, 128, 256],
                 depths=[4, 4, 4, 4],
                 latent_dims=[256, 256, 256, 256]):
        super().__init__()

        self.encoder = VMamba(
                in_channels,
                C=C,
                depths=depths,
                latent_dims=latent_dims,
                return_intermediate=True
        )
        self.decoder = ChangeFormerDecoder(
                num_classes=num_classes,
                C=C,
                embed_dims=latent_dims[3]
        )
    def forward(self, x1, x2):
        x1s = self.encoder(x1)
        x2s = self.encoder(x2)
        for i, x in enumerate(x1s):
            if torch.isnan(x).any():
                print(f"x1s index {i} has NaN")
        for i, x in enumerate(x2s):
            if torch.isnan(x).any():
                print(f"x2s index {i} has NaN")
        return self.decoder(x1s, x2s)

def test_custom_model():
    N, C, W, H = 12, 3, 64, 64
    x1 = torch.rand(N, C, W, H)
    x2 = torch.rand(N, C, W, H)
    cm = CustomModel(3, num_classes=2)
    res = cm(x1, x2)
    assert res.shape == (N, 2, W, H), res.shape
    print(f"Custom Model assert pass! Elapsed time: {time.time()-start_time:.4f}s")


if __name__ == "__main__":
    test_conversion_round_trip()
    test_cross_scan()
    test_s6_block()
    test_crossmerge()
    test_ss2d_core()
    test_ss2d_block()
    test_vss_block()
    test_vmamba()
    test_custom_model()

    print("All asserts pass.")

