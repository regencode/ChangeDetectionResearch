from torch import nn
import einops as ein
from .ChangeFormer import DecoderTransformer_v3
from .ChangeFormerBaseNetworks import UpsampleConvLayer, ResidualBlock, ConvLayer

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

        self.ln = nn.ModuleList(
            nn.LayerNorm(in_chans) for in_chans in in_channels
        )

    def layer_norm(self, x, depth):
        B, C, H, W = x.shape
        x = ein.rearrange(x, "b c h w -> b (h w) c")
        x = self.ln[depth](x)
        return ein.rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

    def forward(self, inputs1, inputs2):
        # Normalize each inputs1 and inputs2
        inputs1 = [self.layer_norm(inputs1[i], i) for i in range(len(inputs1))]
        inputs2 = [self.layer_norm(inputs2[i], i) for i in range(len(inputs2))]
        return super().forward(inputs1, inputs2)
