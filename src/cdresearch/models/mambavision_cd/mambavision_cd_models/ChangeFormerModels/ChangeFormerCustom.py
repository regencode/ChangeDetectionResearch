import torch
from torch import nn
import einops as ein
from .ChangeFormer import DecoderTransformer_v3, EncoderTransformer_v3
from .ChangeFormerBaseNetworks import UpsampleConvLayer, ResidualBlock, ConvLayer

# Transformer Decoder
class MLPCustom(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = self.proj(x)
        return x

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

#class EncoderTransformerCustom(EncoderTransformer_v3):
#    def __init__(self, img_size=256, patch_size=16, in_chans=3, num_classes=2, embed_dims=[64, 128, 256, 512],
#                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
#                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
#                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
#        super().__init__(
#                 img_size, patch_size, in_chans, num_classes, embed_dims,
#                 num_heads, mlp_ratios, qkv_bias, qk_scale, drop_rate,
#                 attn_drop_rate, drop_path_rate, norm_layer,
#                 depths, sr_ratios
#        )
#        self.num_classes = num_classes
#        self.depths = depths
#
#
#        # main  encoder
#        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
#        cur = 0
#        self.block1 = nn.ModuleList([Block(
#            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
#            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
#            sr_ratio=sr_ratios[0])
#            for i in range(depths[0])])
#        self.norm1 = norm_layer(embed_dims[0])
#        # intra-patch encoder
#        self.patch_block1 = nn.ModuleList([Block(
#            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
#            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
#            sr_ratio=sr_ratios[0])
#            for i in range(1)])
#        self.pnorm1 = norm_layer(embed_dims[1])
#        # main  encoder
#        cur += depths[0]
#        self.block2 = nn.ModuleList([Block(
#            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
#            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
#            sr_ratio=sr_ratios[1])
#            for i in range(depths[1])])
#        self.norm2 = norm_layer(embed_dims[1])
#        # intra-patch encoder
#        self.patch_block2 = nn.ModuleList([Block(
#            dim=embed_dims[2], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
#            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
#            sr_ratio=sr_ratios[1])
#            for i in range(1)])
#        self.pnorm2 = norm_layer(embed_dims[2])
#        # main  encoder
#        cur += depths[1]
#        self.block3 = nn.ModuleList([Block(
#            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
#            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
#            sr_ratio=sr_ratios[2])
#            for i in range(depths[2])])
#        self.norm3 = norm_layer(embed_dims[2])
#        # intra-patch encoder
#        self.patch_block3 = nn.ModuleList([Block(
#            dim=embed_dims[3], num_heads=num_heads[1], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
#            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
#            sr_ratio=sr_ratios[2])
#            for i in range(1)])
#        self.pnorm3 = norm_layer(embed_dims[3])
#        # main  encoder
#        cur += depths[2]
#        self.block4 = nn.ModuleList([Block(
#            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
#            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
#            sr_ratio=sr_ratios[3]) for i in range(depths[3])])
#        self.norm4 = norm_layer(embed_dims[3])
#
#        self.apply(self._init_weights)


class MultiLevelFuse(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.layer3 = nn.Sequential(
            ResidualBlockCustom(embedding_dim*2, embedding_dim),
        )
        self.layer2 = nn.Sequential(
            ResidualBlockCustom(embedding_dim*2, embedding_dim),
        )
        self.layer1 = nn.Sequential(
            ResidualBlockCustom(embedding_dim*2, embedding_dim),
        )
    def forward(self, x):
        # x is c4, c3, c2, c1 concatenated at dim=1
        N, C, H, W = x.shape 
        x4, x3, x2, x1 = x.reshape(4, N, -1, H, W)
        _c3 = torch.cat([x4, x3], dim=1)
        _c2 = torch.cat([_c3, x2], dim=1)
        _c1 = torch.cat([_c2, x1], dim=1)
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
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = nn.Sequential(
                ToSequenceForm(),
                MLPCustom(input_dim=c4_in_channels, embed_dim=self.embedding_dim),
        )
        self.linear_c3 = nn.Sequential(
                ToSequenceForm(),
                MLPCustom(input_dim=c3_in_channels, embed_dim=self.embedding_dim),
        )
        self.linear_c2 = nn.Sequential(
                ToSequenceForm(),
                MLPCustom(input_dim=c2_in_channels, embed_dim=self.embedding_dim),
        )
        self.linear_c1 = nn.Sequential(
                ToSequenceForm(),
                MLPCustom(input_dim=c1_in_channels, embed_dim=self.embedding_dim),
        )
        # original is very lossy, original fuses 4*embed_dims -> embed_dims
        self.linear_fuse = MultiLevelFuse(self.embedding_dim)

        #self.dense_2x   = nn.Sequential( ResidualBlockCustom(self.embedding_dim))
        #self.dense_1x   = nn.Sequential( ResidualBlockCustom(self.embedding_dim))
    def forward(self, inputs1, inputs2):
        # inputs are normalize with linear_c1 to c4
        return super().forward(inputs1, inputs2)
