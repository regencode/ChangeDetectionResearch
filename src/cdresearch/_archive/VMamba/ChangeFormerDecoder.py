import torch
from torch import nn
import time
import torch.nn.functional as F
import einops as ein

start_time = time.time()

class DifferenceModule(nn.Module):
    def __init__(self, in_channels, embed_dim=256):
        super().__init__()
        self.proj_pre = nn.Conv2d(in_channels, embed_dim, kernel_size=(1, 1))
        self.proj_post = nn.Conv2d(in_channels, embed_dim, kernel_size=(1, 1))

        self.conv1 = nn.Conv2d(embed_dim*2, embed_dim*2, kernel_size=(3, 3), padding=1, padding_mode="reflect")
        self.bn = nn.BatchNorm2d(num_features=embed_dim*2)
        self.conv2 = nn.Conv2d(embed_dim*2, embed_dim, kernel_size=(3, 3), padding=1, padding_mode="reflect")

    def forward(self, x_pre, x_post):
        x_pre = self.proj_pre(x_pre)
        x_post = self.proj_post(x_post)
        x = torch.cat([x_pre, x_post], dim=1) # concat along dim
        x = F.relu(self.bn(self.conv1(x)));
        return F.relu(self.conv2(x))

def test_difference_module():
    N, C, W, H = 1, 32, 128, 128
    diff = DifferenceModule(C*2, C)
    x1 = torch.rand(N, C, W, H)
    x2 = torch.rand(N, C, W, H)
    res = diff(x1, x2)
    assert res.shape == (N, C, W, H), res.shape
    print(f"DifferenceModule assert passed | Elapsed time: {time.time() - start_time:.4f}s")

class MLPUpsampler(nn.Module):
    def __init__(self, in_channels, embed_dims=256):
        super().__init__()
        self.embed_dims = embed_dims
        self.linear = nn.Conv2d(in_channels, embed_dims, kernel_size=1)
        self.bn = nn.BatchNorm2d(embed_dims)
        
    def forward(self, x, output_size):
        x = self.linear(x);
        x = F.interpolate(x, size=output_size, mode='nearest')
        return x

class MLPFusion(nn.Module):
    def __init__(self, embed_dims=64):
        super().__init__()
        self.linear = nn.Conv2d(embed_dims*4, embed_dims, kernel_size=1)
        self.bn = nn.BatchNorm2d(embed_dims)

    def forward(self, x1, x2, x3, x4):
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.bn(self.linear(x))

def test_mlp_upsampler_and_fusion():
    N, C, W, H = 1, 8, 128, 128
    F1 = torch.rand(N, C, W, H)
    F2 = torch.rand(N, C*2, W//2, H//2)
    F3 = torch.rand(N, C*4, W//4, H//4)
    F4 = torch.rand(N, C*8, W//8, H//8)

    mlp_up1 = MLPUpsampler(C, embed_dims=16)
    mlp_up2 = MLPUpsampler(C*2, embed_dims=16)
    mlp_up3 = MLPUpsampler(C*4, embed_dims=16)
    mlp_up4 = MLPUpsampler(C*8, embed_dims=16)

    res1 = mlp_up1(F1, output_size=(W, H))
    res2 = mlp_up2(F2, output_size=(W, H))
    res3 = mlp_up3(F3, output_size=(W, H))
    res4 = mlp_up4(F4, output_size=(W, H))

    assert res1.shape == (N, 16, W, H), res1.shape
    assert res2.shape == (N, 16, W, H), res2.shape
    assert res3.shape == (N, 16, W, H), res3.shape
    assert res4.shape == (N, 16, W, H), res4.shape

    print(f"MLP Upsampler assert passed | Elapsed time: {time.time() - start_time:.4f}s")

    fuse = MLPFusion(embed_dims=16)
    res = fuse(res1, res2, res3, res4)

    assert res.shape == (N, 16, W, H), res.shape
    print(f"MLP Fusion assert passed | Elapsed time: {time.time() - start_time:.4f}s")


class ConvUpsampleAndClassify(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dims=256):
        super().__init__()
        self.out_channels = out_channels

        # Diverge from original implementation, original implementation 
        # uses single Transposed Conv with kernel_size=4, stride=4 
        # to quadruple spatial dims

        self.conv1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=4, stride=2, padding=1)
        self.dense = nn.Sequential(
                nn.Conv2d(in_channels, embed_dims, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(embed_dims, in_channels, kernel_size=3, padding=1)
        )
        self.conv2 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=4, stride=2, padding=1)
        self.conv_classify = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        N, C, W, H = x.shape
        x = self.conv1(x)
        assert x.shape == (N, C, W*2, H*2), x.shape
        x1 = self.dense(x)
        assert x1.shape == (N, C, W*2, H*2), x1.shape
        x = self.conv2(x + x1)
        assert x.shape == (N, C, W*4, H*4), x.shape
        class_logits = self.conv_classify(x)
        return class_logits

def test_conv_up_and_classify():
    N, C, W, H = 1, 16, 64, 64
    up_class = ConvUpsampleAndClassify(C, 3)
    x = torch.rand(N, C, W, H)
    res = up_class(x)
    assert res.shape == (N, 3, W*4, H*4), res.shape
    print(f"Conv Upsample and Classify assert passed | Elapsed time: {time.time() - start_time:.4f}s")

class ChangeFormerDecoder(nn.Module):
    def __init__(self, num_classes,
                 C=[64, 128, 256, 512], 
                 embed_dims=256):
        super().__init__()
        self.diff1 = DifferenceModule(C[0], embed_dims)
        self.diff2 = DifferenceModule(C[1], embed_dims)
        self.diff3 = DifferenceModule(C[2], embed_dims)
        self.diff4 = DifferenceModule(C[3], embed_dims)

        self.mlp_up1 = MLPUpsampler(embed_dims, embed_dims=embed_dims)
        self.mlp_up2 = MLPUpsampler(embed_dims, embed_dims=embed_dims)
        self.mlp_up3 = MLPUpsampler(embed_dims, embed_dims=embed_dims)
        self.mlp_up4 = MLPUpsampler(embed_dims, embed_dims=embed_dims)

        self.mlp_fusion = MLPFusion(embed_dims)
        self.upsample_and_classify = ConvUpsampleAndClassify(embed_dims, num_classes)
    
    def forward(self, x1s, x2s, return_intermediate=False):
        x1_1, x1_2, x1_3, x1_4 = x1s
        x2_1, x2_2, x2_3, x2_4 = x2s

        F4 = self.diff4(x1_4, x2_4)
        F3 = self.diff3(x1_3, x2_3) + F.interpolate(F4, scale_factor=2, mode="bilinear")
        F2 = self.diff2(x1_2, x2_2) + F.interpolate(F3, scale_factor=2, mode="bilinear")
        F1 = self.diff1(x1_1, x2_1) + F.interpolate(F2, scale_factor=2, mode="bilinear")

        up1 = self.mlp_up1(F1, output_size=x1_1.shape[-2:])
        up2 = self.mlp_up2(F2, output_size=x1_1.shape[-2:])
        up3 = self.mlp_up3(F3, output_size=x1_1.shape[-2:])
        up4 = self.mlp_up4(F4, output_size=x1_1.shape[-2:])
        
        fused = self.mlp_fusion(up1, up2, up3, up4)
        out = self.upsample_and_classify(fused)

        if return_intermediate:
           return out, dict(F1=F1, F2=F2, F3=F3, F4=F4, up1=up1, up2=up2, up3=up3, up4=up4, fused=fused) 
        else:
            return out

