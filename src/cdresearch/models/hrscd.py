import torch
from torch import nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.same = in_channels == out_channels

        self.conv1 = nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=out_channels, padding=1)  
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=out_channels, out_channels=out_channels, padding=1)  
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

        self.shortcut = nn.Identity() if self.same else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.bn2(self.conv2(x))
        x += residual
        return self.relu(x)
        


class DownsampleResBlock(ResBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        self.downsample = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x = super().forward(x)
        return self.downsample(x), x

class UpsampleResBlock(ResBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        self.upsample = nn.ConvTranspose2d(
                in_channels=out_channels, out_channels=out_channels, 
                kernel_size=(2, 2), stride=2
        )

    def forward(self, x, *args):
        if args:
            x_cat = args[0]
            x = torch.cat([x, x_cat], dim=1)
        x = super().forward(x)
        return self.upsample(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_channels=1):
        super().__init__()
        self.res1 = ResBlock(in_channels, in_channels*expand_channels)
        self.res2 = ResBlock(in_channels*expand_channels, in_channels*expand_channels)
        self.res3 = DownsampleResBlock(in_channels*expand_channels, out_channels)
    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x_down, x = self.res3(x)
        return x_down, x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_channels=1):
        super().__init__()
        self.res1 = ResBlock(in_channels, in_channels*expand_channels)
        self.res2 = ResBlock(in_channels*expand_channels, in_channels*expand_channels)
        self.res3 = UpsampleResBlock(in_channels*expand_channels, out_channels)
    def forward(self, x, *args):
        for x_cat in args:
            x = self.res1(torch.cat([x, x_cat], dim=1))
        x = self.res2(x)
        x = self.res3(x)
        return x

class HRSCD_Encoder(nn.Module):
    def __init__(self, in_channels, expand_enc_channels=1):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, 8, expand_enc_channels)
        self.enc2 = EncoderBlock(8, 16, expand_enc_channels)
        self.enc3 = EncoderBlock(16, 32, expand_enc_channels)
        self.enc4 = EncoderBlock(32, 64, expand_enc_channels)
        self.enc5 = EncoderBlock(64, 128, expand_enc_channels)
        self.final_enc = DecoderBlock(128, 128, 2)

    def forward(self, x):
        x, x1 = self.enc1(x)
        x, x2 = self.enc2(x)
        x, x3 = self.enc3(x)
        x, x4 = self.enc4(x)
        x, x5 = self.enc5(x)
        x6 = self.final_enc(x)
        return x1, x2, x3, x4, x5, x6

class HRSCD_Decoder(nn.Module):
    def __init__(self, out_channels, expand_dec_channels=1):
        super().__init__()
        self.dec5 = DecoderBlock(128, 64, expand_dec_channels)
        self.dec4 = DecoderBlock(64, 32, expand_dec_channels)
        self.dec3 = DecoderBlock(32, 16, expand_dec_channels)
        self.dec2 = DecoderBlock(16, 8, expand_dec_channels)
        self.dec1 = DecoderBlock(8, 8*(expand_dec_channels**2), expand_dec_channels)
        self.final_layer = nn.Sequential(
            ResBlock(8*(expand_dec_channels**2), 8*(expand_dec_channels**3)),
            nn.Conv2d(kernel_size=(1, 1), in_channels=8*(expand_dec_channels**3), out_channels=out_channels)
        )

    def forward(self, x1, x2, x3, x4, x5, x6):
        out = self.dec5(x6, x5)
        out = self.dec4(out, x4)
        out = self.dec3(out, x3)
        out = self.dec2(out, x2)
        out = self.dec1(out, x1)
        return self.final_layer(out)


class HRSCD_str4(nn.Module):
    def __init__(self, in_channels, out_channels, expand_enc_channels=1, expand_dec_channels=1):
        super().__init__()
        self.encoder_cd = HRSCD_Encoder(in_channels*2, expand_enc_channels)
        self.decoder_cd = HRSCD_Decoder(2, expand_dec_channels)

        self.encoder_lcm = HRSCD_Encoder(in_channels, expand_enc_channels)
        self.decoder_lcm = HRSCD_Decoder(out_channels, expand_dec_channels)
    
    def forward(self, x1, x2):
        '''
        only call this if the model is already trained.
        else please train each part for better performance (LCM, then CD)
        '''
        x1s = self.encoder_lcm(x1)
        x2s = self.encoder_lcm(x2)

        out_1 = self.decoder_lcm(x1)
        out_2 = self.decoder_lcm(x2)

        x_diff_s = [torch.abs(x1 - x2) for x1, x2 in zip(x1s, x2s)] # absolute difference

        x12s = self.encoder_cd(torch.cat([x1, x2], dim=1))

        x_12_diff_s = [x_diff_s[5], *[torch.cat([x12, x_diff], dim=1) for x12, x_diff in zip(x12s[:5:], x_diff_s[:5:])]]
        # leave deepmost output un-cat with x_diff
        out_binary = self.decoder_cd(x_12_diff_s)

        return out_1, out_2, out_binary
        




