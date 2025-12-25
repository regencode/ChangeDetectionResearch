import torch
from torch import nn
import einops as ein
from .ChangeFormer import DecoderTransformer_v3, MLP
from .ChangeFormerBaseNetworks import UpsampleConvLayer, ResidualBlock, ConvLayer

class ResidualBlockCustom(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.norm = nn.GroupNorm(32, channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        out = self.norm(out)
        return out

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
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = nn.Sequential(
                nn.LayerNorm(c4_in_channels),
                MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim),
        )
        self.linear_c3 = nn.Sequential(
                nn.LayerNorm(c3_in_channels),
                MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim),
        )
        self.linear_c2 = nn.Sequential(
                nn.LayerNorm(c2_in_channels),
                MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim),
        )
        self.linear_c1 = nn.Sequential(
                nn.LayerNorm(c1_in_channels),
                MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim),
        )

        #self.dense_2x   = nn.Sequential( ResidualBlockCustom(self.embedding_dim))
        #self.dense_1x   = nn.Sequential( ResidualBlockCustom(self.embedding_dim))
    def forward(self, inputs1, inputs2):
        # inputs are normalize with linear_c1 to c4
        return super().forward(inputs1, inputs2)
