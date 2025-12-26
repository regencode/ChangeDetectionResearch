import torch
from torch import nn
import einops as ein
from .ChangeFormer import DecoderTransformer_v3, EncoderTransformer_v3
from .ChangeFormerBaseNetworks import UpsampleConvLayer, ResidualBlock, ConvLayer
from ..mamba_vision import Block

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


class EncoderTransformer_v3(nn.Module):
    def __init__(self, img_size=256, patch_size=3, in_chans=3, num_classes=2, embed_dims=[32, 64, 128, 256],
                 num_heads=[2, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 3, 6, 18], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes    = num_classes
        self.depths         = depths
        self.embed_dims     = embed_dims

        # patch embedding definitions
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=patch_size, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=patch_size, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=patch_size, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # Stage-1 (x1/4 scale)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1= MambaVisionLayer(dim=int(dim * 2 ** i),
                                     depth=depths[i],
                                     num_heads=num_heads[i],
                                     window_size=window_size[i],
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     conv=conv,
                                     drop=drop_rate,
                                     attn_drop=attn_drop_rate,
                                     drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                     downsample=(i < 3),
                                     layer_scale=layer_scale,
                                     layer_scale_conv=layer_scale_conv,
                                     transformer_blocks=list(range(depths[i]//2+1, depths[i])) if depths[i]%2!=0 else list(range(depths[i]//2, depths[i])),
                                     )
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        
        # Stage-2 (x1/8 scale)
        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
       
       # Stage-3 (x1/16 scale)
        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        
        # Stage-4 (x1/32 scale)
        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

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
        self.linear_fuse = nn.Sequential(ResidualBlock(self.embedding_dim*4), 
                                         nn.Conv2d(self.embedding_dim*4, self.embedding_dim, kernel_size=1),
                                         nn.BatchNorm2d(self.embedding_dim)
        )


        #self.dense_2x   = nn.Sequential( ResidualBlockCustom(self.embedding_dim))
        #self.dense_1x   = nn.Sequential( ResidualBlockCustom(self.embedding_dim))
    def forward(self, inputs1, inputs2):
        # inputs are normalize with linear_c1 to c4
        return super().forward(inputs1, inputs2)
