from torch import nn
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
