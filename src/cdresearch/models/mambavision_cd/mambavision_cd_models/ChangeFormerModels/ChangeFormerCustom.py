import torch
from torch import nn
import einops as ein
from .ChangeFormer import DecoderTransformer_v3, EncoderTransformer_v3
from .ChangeFormerBaseNetworks import UpsampleConvLayer, ResidualBlock, ConvLayer
from ..mamba_vision import Block


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

class ResidualBlockCustom(ResidualBlock):
    def __init__(self, channels, out_channels):
        super().__init__(channels)
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out

class MultiLevelFuse(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.layer3 = nn.Sequential(
            nn.Conv2d(embedding_dim*2, embedding_dim, 3, 1, 1),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(embedding_dim*2, embedding_dim, 3, 1, 1),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(embedding_dim*2, embedding_dim, 3, 1, 1),
            nn.ReLU(),
        )
    def forward(self, x):
        # x is c4, c3, c2, c1 concatenated at dim=1
        N, C, H, W = x.shape 
        x4, x3, x2, x1 = x.reshape(4, N, -1, H, W)
        _c3 = self.layer3(torch.cat([x4, x3], dim=1))
        _c2 = self.layer2(torch.cat([_c3, x2], dim=1))
        _c1 = self.layer1(torch.cat([_c2, x1], dim=1))
        return _c1

class DecoderTransformerCustom(DecoderTransformer_v3):
    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=True, 
                    in_channels = [32, 64, 128, 256], embedding_dim= 64, output_nc=2, 
                    decoder_softmax = False, feature_strides=[2, 4, 8, 16], 
                    final_upsample=[True, True] # custom
                 ):
        super().__init__(input_transform, in_index, align_corners, 
                         in_channels, embedding_dim, output_nc, 
                         decoder_softmax, feature_strides)
        self.final_upsample = final_upsample
        if not final_upsample[0]:
            self.convd2x = nn.Identity()
        if not final_upsample[1]:
            self.convd1x = nn.Identity()


        #self.dense_2x   = nn.Sequential( ResidualBlockCustom(self.embedding_dim))
        #self.dense_1x   = nn.Sequential( ResidualBlockCustom(self.embedding_dim))
    def forward(self, inputs1, inputs2):
        # inputs are normalize with linear_c1 to c4
        return super().forward(inputs1, inputs2)
