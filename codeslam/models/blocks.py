###################################################################
### Code modified from https://github.com/milesial/Pytorch-UNet ###
###################################################################

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, linear=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.Identity() if linear else nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Identity() if linear else nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConvCat(nn.Module):
    """Concat then double conv"""
    def __init__(self, in_channels, out_channels, mid_channels=None, linear=False):
        super().__init__()
        self.conv = DoubleConv(in_channels*2, out_channels, mid_channels=mid_channels, linear=linear)
    
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class DownCat(nn.Module):
    """Downscaling with maxpool, conv with increase of channels,
       concat with external feature map, then conv with reduction of channels
       since concat doubles channels
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x1, x2):
        x1 = self.down(x1)
        x1 = self.conv1(x1)

        x = torch.cat([x1, x2], dim=1)
        
        return self.conv2(x)
        

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, linear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            # TODO: Fix so bilinear doesn't need an extra Conv layer
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1) # Only for reducing channels
            )
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, linear=linear)

    def forward(self, x1, x2):
        x1 = self.up(x1) # (N, 256, 12, 16)
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2) # "Deconvolution" for the last layer

    def forward(self, x):
        return self.up(x)