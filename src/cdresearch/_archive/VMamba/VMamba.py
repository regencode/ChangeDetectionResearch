import torch
from torch import nn
import time
import torch.nn.functional as F
import einops as ein
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

start_time = time.time()

class ToSequenceForm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if x.ndim == 3: return x # already sequence
        return ein.rearrange(x, "b c h w -> b (h w) c")

class ToImageForm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        '''
        assume image has equal width and height, and sequence length is a perfect square
        '''
        if x.ndim == 4: return x # already image

        B, L, D = x.shape
        H = W = int(L ** 0.5)
        assert H * W == L, "L must be a perfect square"
        return ein.rearrange(x, "b (h w) d -> b d h w", h=H, w=W)

def test_conversion_round_trip():
    to_seq = ToSequenceForm()
    to_img = ToImageForm()
    N, C, W, H = 2, 8, 32, 32
    x = torch.rand(N, C, W, H)
    assert torch.equal(x, to_img(to_seq(x))), "conversion round trip img -> seq -> img fail"

    x = torch.rand(N, W*H, C)
    assert torch.equal(x, to_seq(to_img(x))), "conversion round trip seq -> img -> seq fail" 
    print(f"Conversion round trip assert pass | Time elapsed: {(time.time() - start_time):.4f}s")    

def cross_scan(x: torch.Tensor):
    B, C, H, W = x.shape
    # scans
    scan_1 = ein.rearrange(x, "b c h w -> b (h w) c")
    scan_2 = ein.rearrange(x, "b c h w -> b (w h) c", h=H, w=W)
    scan_3 = scan_1.flip(1)
    scan_4 = scan_2.flip(1)

    return scan_1, scan_2, scan_3, scan_4

def test_cross_scan():
    N, C, W, H = 1, 3, 32, 32
    x = torch.rand(N, C, W, H)
    linear_shape = (N, W*H, C)
    res = cross_scan(x)
    for idx in range(1, len(res)):
        assert res[idx].shape == (N, W*H, C), f"res shape ({res[idx].shape}) != ({linear_shape})"
        assert res[idx].shape == res[idx-1].shape, "scan result shapes does not match"
    print(f"Cross scan assert pass | Time elapsed: {(time.time() - start_time):.4f}s")    


class S6Block(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim=512, h0 = None):
        super().__init__()

        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.h0 = 0 if h0 is None else h0
        self.x_proj = nn.Linear(in_channels, latent_dim)
        self.bc_proj = nn.Linear(in_channels, latent_dim*2)
        self.del_proj = nn.Linear(in_channels, latent_dim)
        
        self.y_proj = nn.Linear(latent_dim, out_channels)

        self.D = nn.Identity()

    
    def forward(self, x: torch.Tensor):
        x_proj = self.x_proj(x)
        BC = self.bc_proj(x)
        delta = self.del_proj(x)
        B, C = BC.chunk(2, dim=-1)
        
        A_bar = torch.exp(-F.softplus(delta)) # exponential decay, exponent must be small negative therefore softplus is used
        B_bar = (1 - A_bar) * B

        assert A_bar.shape == B_bar.shape, f"A_bar.shape ({A_bar.shape}) does not match B_bar.shape ({B.shape})."
        assert B_bar.shape == C.shape, f"B_bar.shape ({B_bar.shape}) does not match C.shape ({C.shape})."
        selective_scan_fn(x, delta, A_bar, B_bar, C)

        return self.y_proj(y)

def test_s6_block():
    N, C, W, H = 2, 3, 32, 32
    x = torch.rand(N, W*H, C)
    s6 = S6Block(3, 32, 128)
    res = s6(x)
    assert res.shape == (N, W*H, 32), res.shape
    print(f"S6 Block assert pass | Time elapsed: {(time.time() - start_time):.4f}s")    



class CrossMerge(nn.Module):
    def __init__(self, y_in_channels, out_channels):
        super().__init__()
        self.merge_proj = nn.Linear(y_in_channels*4, y_in_channels)
        self.gate_proj = nn.Linear(y_in_channels*4, y_in_channels)
        self.out_proj = nn.Linear(y_in_channels, out_channels)

    def forward(self, y1, y2, y3, y4):
        y_cat = torch.cat([y1, y2, y3, y4], dim=-1) # cat on channel dim
        y_merged = self.merge_proj(y_cat)
        gate = torch.sigmoid(self.gate_proj(y_cat)) 
        return self.out_proj(y_merged * gate)

def test_crossmerge():
    N, C, W, H = 2, 8, 32, 32
    x1 = torch.rand(N, W*H, C)
    x2 = torch.rand(N, W*H, C)
    x3 = torch.rand(N, W*H, C)
    x4 = torch.rand(N, W*H, C)

    cm = CrossMerge(8, 16)
    res = cm(x1, x2, x3, x4)
    assert res.shape == (N, W*H, 16), res.shape
    print(f"Cross Merge assert pass | Time elapsed: {(time.time() - start_time):.4f}s")    

class SS2DCore(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim=256):
        super().__init__()
        self.s6 = S6Block(in_channels, out_channels=latent_dim, latent_dim=latent_dim)
        self.cross_merge = CrossMerge(latent_dim, out_channels)

    def forward(self, x):
        _, _, H, W = x.shape
        x1, x2, x3, x4 = cross_scan(x)
        x_merged = self.cross_merge(
            self.s6(x1),
            self.s6(x2),
            self.s6(x3),
            self.s6(x4)
        )
        x_merged = ein.rearrange(x_merged, "b (h w) c -> b c h w", h=H, w=W) # to image form
        return x_merged

def test_ss2d_core():
    N, C, W, H = 2, 8, 32, 32
    x = torch.rand(N, C, W, H)

    ss2d = SS2DCore(8, 16)
    res = ss2d(x)
    assert res.shape == (N, 16, W, H), res.shape
    print(f"SS2D Core assert pass | Time elapsed: {(time.time() - start_time):.4f}s")    

class SS2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim=64):
        super().__init__()
        self.layers = nn.Sequential(
            ToImageForm(),
            nn.Conv2d(in_channels, latent_dim, kernel_size=1),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1, groups=latent_dim),
            nn.SiLU(),
            SS2DCore(latent_dim, latent_dim, latent_dim=latent_dim),
            ToSequenceForm(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, out_channels),
            ToImageForm()
        )

    def forward(self, x):
        '''
        Assume Image form
        '''
        return self.layers(x)

def test_ss2d_block():
    N, C, W, H = 2, 8, 32, 32
    x = torch.rand(N, C, W, H)

    ss2d = SS2DBlock(8, 16, latent_dim=64)
    res = ss2d(x)
    assert res.shape == (N, 16, W, H), res.shape
    print(f"SS2D Block assert pass | Time elapsed: {(time.time() - start_time):.4f}s")    

class VSSBlock(nn.Module):
    def __init__(self, in_channels, latent_dim=256):
        super().__init__()
        self.to_seq = ToSequenceForm()
        self.to_img = ToImageForm()
        self.ln1 = nn.LayerNorm(in_channels)
        self.ss2d_block = SS2DBlock(in_channels, in_channels, latent_dim=latent_dim)
        self.ln2 = nn.LayerNorm(in_channels)
        self.ffn = nn.Sequential(
            self.to_img,
            nn.Conv2d(in_channels, latent_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(latent_dim, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.to_img(x) + self.ss2d_block(self.ln1(self.to_seq(x)))
        x1 = self.to_img(self.ln2(self.to_seq(x)))
        return x + self.ffn(x1)

def test_vss_block():
    N, C, W, H = 2, 8, 32, 32
    x = torch.rand(N, C, W, H)

    vss = VSSBlock(8, latent_dim=64)
    res = vss(x)
    assert res.shape == (N, 8, W, H), res.shape
    print(f"VSS Block assert pass | Time elapsed: {(time.time() - start_time):.4f}s")    


class Downsample(nn.Module):
    def __init__(self, in_channels, embed_dim=256, kernel_size=7, stride=4, padding=0):
        super().__init__()
        self.conv_to_patch = nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.conv_to_patch(x)
        _, _, H, W = x.shape
        x = x.flatten(start_dim=-2).transpose(-2, -1)
        x = self.layernorm(x)
        return x

class VMamba(nn.Module):
    def __init__(self, in_channels, 
                 C=[32, 64, 128, 256],
                 depths=[4, 4, 4, 4],
                 latent_dims=[256, 256, 256, 256],
                 return_intermediate=False):
        super().__init__()
        self.return_intermediate = return_intermediate
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C[0], kernel_size=4, stride=2, padding=1),
            nn.Conv2d(C[0], C[0], kernel_size=4, stride=2, padding=1)
        )
        self.stage1 = nn.Sequential(*[
            VSSBlock(C[0], latent_dim=latent_dims[0])
            for _ in range(depths[0])
        ])
        self.stage2 = nn.Sequential(
            Downsample(C[0], C[1], kernel_size=4, stride=2, padding=1),
            *[
                VSSBlock(C[1], latent_dim=latent_dims[1])
                for _ in range(depths[1])
            ]
        )
        self.stage3 = nn.Sequential(
            Downsample(C[1], C[2], kernel_size=4, stride=2, padding=1),
            *[
                VSSBlock(C[2], latent_dim=latent_dims[2])
                for _ in range(depths[2])
            ]
        )
        self.stage4 = nn.Sequential(
            Downsample(C[2], C[3], kernel_size=4, stride=2, padding=1),
            *[
                VSSBlock(C[3], latent_dim=latent_dims[3])
                for _ in range(depths[3])
            ]
        )

    def forward(self, x):
        x = self.stem(x)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        if self.return_intermediate:
            return x1, x2, x3, x4
        return x4
        

def test_vmamba():
    N, C, W, H = 2, 3, 256, 256
    x = torch.rand(N, C, W, H)

    vmamba = VMamba(3,
                 C=[16, 32, 48, 64],
                 depths=[2, 2, 2, 2],
                 latent_dims=[256, 256, 256, 256])
    res = vmamba(x)
    assert res.shape == (N, 64, W//32, H//32), res.shape
    print(f"VMamba assert pass | Time elapsed: {(time.time() - start_time):.4f}s")    


if __name__ == "__main__":
    test_conversion_round_trip()
    test_cross_scan()
    test_s6_block()
    test_crossmerge()
    test_ss2d_core()
    test_ss2d_block()
    test_vss_block()
    test_vmamba()

    print("All asserts pass.")

