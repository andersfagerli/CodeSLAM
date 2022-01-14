###################################################################
### Code modified from https://github.com/milesial/Pytorch-UNet ###
###################################################################

import torch.nn as nn
import torchvision

from .blocks import DoubleConv, Down, Up, OutConv

class UNet(nn.Module):
    def __init__(self, cfg, bilinear=True):
        super(UNet, self).__init__()
        in_ch = cfg.INPUT.IMAGE_CHANNELS
        out_ch = cfg.OUTPUT.CHANNELS
        e_ch = cfg.MODEL.UNET.ENCODER.OUT_CHANNELS
        d_ch = cfg.MODEL.UNET.DECODER.OUT_CHANNELS

        self.down1 = Down(in_ch,   e_ch[0])  # (1, 16)
        self.down2 = Down(e_ch[0], e_ch[1])  # (16, 32)
        self.down3 = Down(e_ch[1], e_ch[2])  # (32, 64)
        self.down4 = Down(e_ch[2], e_ch[3])  # (64, 128)
        self.down5 = Down(e_ch[3], e_ch[4])  # (128, 256)

        self.d_inc = DoubleConv(d_ch[0], d_ch[0]) # (256, 256)
        self.up1 = Up(d_ch[0], d_ch[1], bilinear) # (256, 128)
        self.up2 = Up(d_ch[1], d_ch[2], bilinear) # (128, 64)
        self.up3 = Up(d_ch[2], d_ch[3], bilinear) # (64, 32)
        self.up4 = Up(d_ch[3], d_ch[4], bilinear) # (32, 16)

        self.out = nn.Sequential(
            OutConv(d_ch[4], out_ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forwards input and returns each feature map in the decoder and final output
        """
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x1_out = self.d_inc(x5)
        x2_out = self.up1(x1_out, x4)
        x3_out = self.up2(x2_out, x3)
        x4_out = self.up3(x3_out, x2)
        x5_out = self.up4(x4_out, x1)
        out = self.out(x5_out)

        return (x1_out, x2_out, x3_out, x4_out, x5_out, out)


class UNetResNet18(nn.Module):
    def __init__(self, cfg, bilinear=True):
        super(UNetResNet18, self).__init__()
        in_ch = cfg.INPUT.IMAGE_CHANNELS
        out_ch = cfg.OUTPUT.CHANNELS
        d_ch = cfg.MODEL.UNET.DECODER.OUT_CHANNELS
        pretrained = cfg.MODEL.UNET.ENCODER.PRETRAINED

        model = torchvision.models.resnet18(pretrained)

        self.down1 = model.conv1    # (3, 64)
        self.block = model.layer1   # (64, 64)
        self.down2 = model.layer2   # (64, 128)
        self.down3 = model.layer3   # (128, 256)
        self.down4 = model.layer4   # (256, 512)

        self.d_inc = DoubleConv(d_ch[0], d_ch[0]) # (512, 512)
        self.up1 = Up(d_ch[0], d_ch[1], bilinear) # (512, 256)
        self.up2 = Up(d_ch[1], d_ch[2], bilinear) # (256, 128)
        self.up3 = Up(d_ch[2], d_ch[3], bilinear) # (128, 64)

        self.out = nn.Sequential(
            OutConv(d_ch[3], out_ch),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forwards input and returns each feature map in the decoder and final output
        """
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x1_out = self.d_inc(x4)
        x2_out = self.up1(x1_out, x3)
        x3_out = self.up2(x2_out, x2)
        x4_out = self.up3(x3_out, x1)

        out = self.out(x4_out)

        return (x1_out, x2_out, x3_out, x4_out, out)