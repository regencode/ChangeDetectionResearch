import torch
from torch import nn

class FC_Double_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=(3, 3))

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class FC_Encoder_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2) # 2x2 maxpool
        self.double_conv = FC_Double_Conv(in_channels, 16)

    def __call__(self, x):
        x = self.maxpool(x)
        x = self.double_conv(x)
        return x

class FC_Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.double_conv1 = FC_Double_Conv(in_channels, 16)
        self.enc_block1 = FC_Encoder_Block(16, 32)
        self.enc_block2 = FC_Encoder_Block(32, 64)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=(3, 3))
        self.enc_block3 = FC_Encoder_Block(64, 128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3))

    def __call__(self, x):
        x1 = self.double_conv1(x)
        x2 = self.enc_block1(x1)
        x3 = self.conv1(self.enc_block2(x2))
        x4 = self.conv2(self.enc_block3(x3))
        return x1, x2, x3, x4

class FC_Decoder_Block1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = in_channels//2
        self.out_channels = out_channels

        self.upconv1 = nn.ConvTranspose2d(self.in_channels, self.mid_channels, kernel_size=(3, 3), stride=1, padding=1)
        self.upconv2 = nn.ConvTranspose2d(self.mid_channels, self.mid_channels, kernel_size=(3, 3), stride=1, padding=1)
        self.upconv3 = nn.ConvTranspose2d(self.mid_channels, self.out_channels, kernel_size=(3, 3), stride=1, padding=1)
        self.upconv4 = nn.ConvTranspose2d(self.out_channels, self.out_channels, kernel_size=(2, 2), stride=2)

    def __call__(self, x):
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)
        return x

class FC_Decoder_Block2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = in_channels//2
        self.out_channels = out_channels

        self.upconv1 = nn.ConvTranspose2d(self.in_channels, self.mid_channels, kernel_size=(3, 3), stride=1, padding=1)
        self.upconv2 = nn.ConvTranspose2d(self.mid_channels, self.out_channels, kernel_size=(3, 3), stride=1, padding=1)
        self.upconv3 = nn.ConvTranspose2d(self.out_channels, self.out_channels, kernel_size=(2, 2), stride=2)

    def __call__(self, x):
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        return x

class FC_Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2) # 2x2 maxpool
        self.upconv1 = nn.ConvTranspose2d(self.in_channels, self.in_channels, kernel_size=(2, 2), stride=2)

        self.decoder_block1 = FC_Decoder_Block1(self.in_channels*2, 64)
        self.decoder_block2 = FC_Decoder_Block1(64*2, 32)
        self.decoder_block3 = FC_Decoder_Block2(32*2, 16)
        self.output_upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=(2, 2), stride=2)
        self.output_upconv2 = nn.ConvTranspose2d(16, self.out_channels, kernel_size=(2, 2), stride=2)

    def __call__(self, x, cat_1, cat_2, cat_3, cat_4):
        x = self.maxpool(x)
        x = torch.cat(self.upconv1(x), cat_1)
        x = torch.cat(self.decoder_block1(x), cat_2)
        x = torch.cat(self.decoder_block2(x), cat_3)
        x = torch.cat(self.decoder_block3(x), cat_4)
        x = self.output_upconv1(x)
        x = self.output_upconv2(x)
        return x


class FC_EF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = FC_Encoder(self.in_channels, 128)
        self.decoder = FC_Decoder(128, self.out_channels)

    def __call__(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        return self.decoder(x4, x4, x3, x2, x1)



